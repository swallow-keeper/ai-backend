from __future__ import annotations

import io
import wave
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from swallow_shared.constants import MAX_DECODE_SECONDS, MAX_UPLOAD_BYTES, SAMPLE_RATE, SEGMENT_SECONDS


class AudioDecodeError(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    if len(data) > MAX_UPLOAD_BYTES:
        raise AudioDecodeError("TOO_LARGE")
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.getnframes()
            raw = handle.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise AudioDecodeError("DECODE_FAILED") from exc
    if width != 2 or channels < 1 or rate <= 0 or frames <= 0:
        raise AudioDecodeError("DECODE_FAILED")
    if frames / rate > MAX_DECODE_SECONDS:
        raise AudioDecodeError("TOO_LONG")
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if channels > 1:
        usable = (pcm.size // channels) * channels
        pcm = pcm[:usable].reshape(-1, channels).mean(axis=1)
    return pcm / 32768.0, rate


def to_32k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == SAMPLE_RATE:
        return np.ascontiguousarray(samples, dtype=np.float32)
    divisor = gcd(sample_rate, SAMPLE_RATE)
    converted = resample_poly(samples, SAMPLE_RATE // divisor, sample_rate // divisor)
    return np.ascontiguousarray(converted, dtype=np.float32)


def _frame_rms(samples: np.ndarray, frame: int) -> np.ndarray:
    count = samples.size // frame
    if count == 0:
        return np.zeros(0, dtype=np.float32)
    shaped = samples[: count * frame].reshape(count, frame)
    return np.sqrt(np.mean(shaped * shaped, axis=1) + 1e-12)


def segment_voice(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, dict[str, float]]:
    # 앞쪽 무음 제거 후 2초, 부족하면 0 패딩
    frame = max(int(0.02 * sample_rate), 1)
    energies = _frame_rms(samples, frame)
    if energies.size == 0:
        segment = np.zeros(int(sample_rate * SEGMENT_SECONDS), dtype=np.float32)
        return segment, _empty_metrics()
    absolute = 0.02
    floor = float(np.percentile(energies, 20))
    voiced_idx = np.flatnonzero(energies >= absolute)
    onset_frame = int(voiced_idx[0]) if voiced_idx.size else 0
    onset = onset_frame * frame
    need = int(sample_rate * SEGMENT_SECONDS)
    raw = samples[onset : onset + need]
    if voiced_idx.size:
        end_frame = onset_frame + max(need // frame, 1)
        in_window = voiced_idx[(voiced_idx >= onset_frame) & (voiced_idx < end_frame)]
        voiced_seconds = float(in_window.size * frame / sample_rate)
    else:
        voiced_seconds = 0.0
    if voiced_idx.size:
        # onset 이전 무음만 소음 바닥으로 사용
        leading = samples[:onset]
        noise_rms = float(np.sqrt(np.mean(leading * leading) + 1e-12)) if leading.size >= frame else None
    else:
        noise_rms = float(np.sqrt(np.mean(samples * samples) + 1e-12))
    signal_rms = float(np.sqrt(np.mean(raw * raw) + 1e-12)) if raw.size else 0.0
    clipping = float(np.mean(np.abs(raw) >= 0.99)) if raw.size else 0.0
    if noise_rms is None:
        snr_db = 40.0
        noise_rms = floor
    else:
        snr_db = float(20.0 * np.log10((signal_rms + 1e-8) / (noise_rms + 1e-8)))
    segment = np.zeros(need, dtype=np.float32)
    segment[: raw.size] = raw
    return segment, {
        "voiced_seconds": voiced_seconds,
        "rms": signal_rms,
        "clipping_ratio": clipping,
        "snr_db": snr_db,
        "noise_rms": float(noise_rms),
    }


def _empty_metrics() -> dict[str, float]:
    return {
        "voiced_seconds": 0.0,
        "rms": 0.0,
        "clipping_ratio": 0.0,
        "snr_db": 0.0,
        "noise_rms": 0.0,
    }
