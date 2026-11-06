from __future__ import annotations

import numpy as np
from scipy.signal import stft

from swallow_shared.constants import MEL_BINS, MEL_FFT, MEL_HOP, MEL_WINDOW, PREEMPHASIS, SAMPLE_RATE

_BANKS: dict[int, np.ndarray] = {}


def preemphasis(samples: np.ndarray, coef: float = PREEMPHASIS) -> np.ndarray:
    if samples.size == 0:
        return samples.astype(np.float32, copy=False)
    out = np.empty(samples.shape, dtype=np.float32)
    out[0] = samples[0]
    out[1:] = samples[1:] - coef * samples[:-1]
    return out


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    cached = _BANKS.get(sample_rate)
    if cached is not None:
        return cached
    # 논문 mel 사양, 32kHz, window 640, hop 320, fft 640, mels 128
    freqs = MEL_FFT // 2 + 1
    points = np.linspace(_hz_to_mel(0.0), _hz_to_mel(sample_rate / 2), MEL_BINS + 2)
    bins = np.floor((MEL_FFT + 1) * _mel_to_hz(points) / sample_rate).astype(int)
    bank = np.zeros((MEL_BINS, freqs), dtype=np.float32)
    for index in range(MEL_BINS):
        left, center, right = int(bins[index]), int(bins[index + 1]), int(bins[index + 2])
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        up = np.arange(left, center)
        up = up[(up >= 0) & (up < freqs)]
        if up.size:
            bank[index, up] = (up - left) / (center - left)
        down = np.arange(center, right)
        down = down[(down >= 0) & (down < freqs)]
        if down.size:
            bank[index, down] = (right - down) / (right - center)
    _BANKS[sample_rate] = bank
    return bank


def power_mel(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    emphasized = preemphasis(samples)
    _, _, spectrum = stft(
        emphasized,
        fs=sample_rate,
        window="hann",
        nperseg=MEL_WINDOW,
        noverlap=MEL_WINDOW - MEL_HOP,
        nfft=MEL_FFT,
        boundary=None,
        padded=False,
        detrend=False,
    )
    power = np.abs(spectrum).astype(np.float32) ** 2
    return mel_filterbank(sample_rate) @ power
