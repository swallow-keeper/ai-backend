import io
import wave

import numpy as np

from swallow_shared.audio import AudioDecodeError, decode_wav, segment_voice, to_32k
from swallow_shared.constants import MEL_BINS, SAMPLE_RATE
from swallow_shared.mel import power_mel
from swallow_shared.quality import prepare_clip


def _sine(seconds: float, rate: int, freq: float = 200.0, amp: float = 0.2) -> np.ndarray:
    time = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return (amp * np.sin(2 * np.pi * freq * time)).astype(np.float32)


def _wav(samples: np.ndarray, rate: int, channels: int = 1) -> bytes:
    if channels == 2:
        stereo = np.stack([samples, samples], axis=1)
        pcm = np.clip(stereo, -1, 1)
    else:
        pcm = np.clip(samples, -1, 1)
    payload = (pcm * 32767).astype(np.int16).tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(payload)
    return buffer.getvalue()


def test_resample_and_downmix_to_32k_mono():
    samples = _sine(1.0, 44_100)
    decoded, rate = decode_wav(_wav(samples, 44_100, channels=2))
    assert rate == 44_100
    converted = to_32k(decoded, rate)
    assert converted.ndim == 1
    assert abs(converted.size - SAMPLE_RATE) < 5


def test_trims_leading_silence_and_pads_to_two_seconds():
    rate = SAMPLE_RATE
    clip = np.concatenate([np.zeros(int(0.4 * rate), dtype=np.float32), _sine(0.8, rate)])
    segment, metrics = segment_voice(clip, rate)
    assert segment.size == SAMPLE_RATE * 2
    assert metrics["voiced_seconds"] == 0.8
    assert float(np.max(np.abs(segment[: int(0.2 * rate)]))) > 0.05


def test_power_mel_matches_paper_shape():
    mel = power_mel(_sine(2.0, SAMPLE_RATE))
    assert mel.shape[0] == MEL_BINS
    assert mel.shape[1] > 10
    assert np.isfinite(mel).all()
    assert mel.min() >= 0


def test_clean_vowel_passes_quality_gate():
    report = prepare_clip(_wav(_sine(2.0, SAMPLE_RATE), SAMPLE_RATE))
    assert report.passed, report.reasons
    assert report.segment.size == SAMPLE_RATE * 2


def test_short_quiet_clipped_and_noise_fail_with_reason_codes():
    short = prepare_clip(_wav(_sine(0.2, SAMPLE_RATE), SAMPLE_RATE))
    assert "VOICED_TOO_SHORT" in short.reasons
    quiet = prepare_clip(_wav(_sine(2.0, SAMPLE_RATE, amp=0.001), SAMPLE_RATE))
    assert "TOO_QUIET" in quiet.reasons
    clipped = prepare_clip(_wav(_sine(2.0, SAMPLE_RATE, amp=0.999), SAMPLE_RATE))
    assert "CLIPPING" in clipped.reasons
    noise = np.random.default_rng(0).normal(0, 0.15, SAMPLE_RATE * 2).astype(np.float32)
    noisy = prepare_clip(_wav(noise, SAMPLE_RATE))
    assert "NOT_SUSTAINED_VOWEL" in noisy.reasons


def test_level_shift_is_warning_not_failure():
    report = prepare_clip(
        _wav(_sine(2.0, SAMPLE_RATE, amp=0.2), SAMPLE_RATE),
        baseline_level_db=-6.0,
        baseline_noise_db=-80.0,
    )
    assert report.passed
    assert "LEVEL_SHIFT" in report.warnings or "NOISE_FLOOR_SHIFT" in report.warnings


def test_reject_non_wav_bytes():
    try:
        decode_wav(b"not-a-wav")
    except AudioDecodeError as exc:
        assert exc.code == "DECODE_FAILED"
    else:
        raise AssertionError("expected decode failure")
