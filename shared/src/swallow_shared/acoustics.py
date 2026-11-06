from __future__ import annotations

import numpy as np

from swallow_shared.constants import SAMPLE_RATE
from swallow_shared.mel import power_mel


def mfcc_mean(samples: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
    log_mel = np.log(power_mel(samples) + 1e-10)
    bins = log_mel.shape[0]
    basis = np.cos(
        np.pi / bins * (np.arange(bins)[:, None] + 0.5) * np.arange(n_mfcc)[None, :]
    )
    coeffs = basis.T @ log_mel
    return coeffs.mean(axis=1).astype(np.float32)


def cpp_mean(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    # 설명용 보조값, 판정에 사용하지 않음
    frame = 2048
    hop = 1024
    if samples.size < frame:
        return 0.0
    window = np.hanning(frame).astype(np.float32)
    peaks: list[float] = []
    min_q = max(int(sample_rate / 400), 1)
    max_q = min(int(sample_rate / 70), frame // 2)
    for start in range(0, samples.size - frame + 1, hop):
        chunk = samples[start : start + frame] * window
        cepstrum = np.fft.irfft(np.log(np.abs(np.fft.rfft(chunk)) + 1e-8))
        zone = cepstrum[min_q:max_q]
        if zone.size == 0:
            continue
        peaks.append(float(np.max(zone) - np.mean(zone)))
    if not peaks:
        return 0.0
    return float(np.mean(peaks))


def acoustic_delta(
    samples: np.ndarray,
    baseline_mfcc: np.ndarray,
    baseline_cpp: float,
) -> dict[str, float]:
    current_mfcc = mfcc_mean(samples)
    current_cpp = cpp_mean(samples)
    return {
        "mfcc_l2": float(np.linalg.norm(current_mfcc - baseline_mfcc)),
        "cpp_delta": float(current_cpp - baseline_cpp),
    }
