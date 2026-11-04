from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from swallow_shared.audio import decode_wav, segment_voice, to_32k
from swallow_shared.constants import SAMPLE_RATE


@dataclass
class QualityConfig:
    min_voiced_seconds: float = 0.5
    min_rms: float = 0.02
    max_clipping_ratio: float = 0.01
    min_snr_db: float = 6.0
    min_voiced_ratio: float = 0.55
    max_f0_cv: float = 0.25
    max_level_delta_db: float = 12.0
    max_noise_floor_delta_db: float = 10.0


@dataclass
class QualityReport:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    voiced_seconds: float = 0.0
    rms: float = 0.0
    clipping_ratio: float = 0.0
    snr_db: float = 0.0
    voiced_ratio: float = 0.0
    f0_cv: float | None = None
    input_level_db: float = -120.0
    noise_floor_db: float = -120.0
    segment: np.ndarray = field(default_factory=lambda: np.zeros(SAMPLE_RATE * 2, dtype=np.float32))


def prepare_clip(
    data: bytes,
    config: QualityConfig | None = None,
    baseline_level_db: float | None = None,
    baseline_noise_db: float | None = None,
) -> QualityReport:
    config = config or QualityConfig()
    samples, rate = decode_wav(data)
    segment, metrics = segment_voice(to_32k(samples, rate), SAMPLE_RATE)
    f0_cv, voiced_ratio = _f0_stats(segment, metrics["voiced_seconds"], SAMPLE_RATE)
    reasons: list[str] = []
    if metrics["voiced_seconds"] < config.min_voiced_seconds:
        reasons.append("VOICED_TOO_SHORT")
    if metrics["rms"] < config.min_rms:
        reasons.append("TOO_QUIET")
    if metrics["clipping_ratio"] > config.max_clipping_ratio:
        reasons.append("CLIPPING")
    if metrics["snr_db"] < config.min_snr_db:
        reasons.append("LOW_SNR")
    if voiced_ratio < config.min_voiced_ratio or f0_cv is None or f0_cv > config.max_f0_cv:
        reasons.append("NOT_SUSTAINED_VOWEL")
    input_level = _db(metrics["rms"])
    noise_floor = _db(metrics["noise_rms"])
    warnings: list[str] = []
    if baseline_level_db is not None and abs(input_level - baseline_level_db) > config.max_level_delta_db:
        warnings.append("LEVEL_SHIFT")
    if baseline_noise_db is not None and abs(noise_floor - baseline_noise_db) > config.max_noise_floor_delta_db:
        warnings.append("NOISE_FLOOR_SHIFT")
    return QualityReport(
        passed=not reasons,
        reasons=reasons,
        warnings=warnings,
        voiced_seconds=metrics["voiced_seconds"],
        rms=metrics["rms"],
        clipping_ratio=metrics["clipping_ratio"],
        snr_db=metrics["snr_db"],
        voiced_ratio=voiced_ratio,
        f0_cv=f0_cv,
        input_level_db=input_level,
        noise_floor_db=noise_floor,
        segment=segment,
    )


def _db(amplitude: float) -> float:
    return float(20.0 * np.log10(amplitude + 1e-8))


def _f0_stats(segment: np.ndarray, voiced_seconds: float, sample_rate: int) -> tuple[float | None, float]:
    usable = int(min(segment.size, max(voiced_seconds, 0.0) * sample_rate))
    region = segment[:usable]
    frame = int(0.04 * sample_rate)
    hop = int(0.02 * sample_rate)
    if region.size < frame:
        return None, 0.0
    pitches: list[float] = []
    starts = range(0, region.size - frame + 1, hop)
    total = 0
    for start in starts:
        total += 1
        pitch = _frame_f0(region[start : start + frame], sample_rate)
        if pitch is not None:
            pitches.append(pitch)
    ratio = (len(pitches) / total) if total else 0.0
    if len(pitches) < 3:
        return None, ratio
    values = np.asarray(pitches, dtype=np.float64)
    cv = float(np.std(values) / (np.mean(values) + 1e-8))
    return cv, ratio


def _frame_f0(frame: np.ndarray, sample_rate: int) -> float | None:
    centered = frame - float(np.mean(frame))
    energy = float(np.dot(centered, centered))
    if energy <= 1e-6:
        return None
    corr = np.correlate(centered, centered, mode="full")
    corr = corr[corr.size // 2 :]
    min_lag = max(int(sample_rate / 400), 1)
    max_lag = min(int(sample_rate / 70), corr.size - 1)
    if max_lag <= min_lag:
        return None
    zone = corr[min_lag:max_lag]
    peak = float(np.max(zone))
    if peak / (corr[0] + 1e-8) < 0.3:
        return None
    lag = min_lag + int(np.argmax(zone))
    return sample_rate / lag
