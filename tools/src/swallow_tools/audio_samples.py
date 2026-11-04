from __future__ import annotations

import io
import wave

import numpy as np


def sine_wav(seconds: float = 2.0, freq: float = 200.0, amp: float = 0.2, rate: int = 32_000) -> bytes:
    time = np.arange(int(seconds * rate), dtype=np.float32) / rate
    samples = amp * np.sin(2 * np.pi * freq * time)
    payload = (np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(payload)
    return buffer.getvalue()
