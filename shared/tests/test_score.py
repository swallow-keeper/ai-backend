from pathlib import Path

import numpy as np
import pytest

from swallow_shared.acoustics import acoustic_delta, cpp_mean, mfcc_mean
from swallow_shared.constants import SAMPLE_RATE
from swallow_shared.embedding import (
    ClassifierHead,
    EfficientATExtractor,
    EmbeddingUnavailable,
    MelPoolExtractor,
    WavLMExtractor,
)
from swallow_shared.registry import load_registry
from swallow_shared.score import (
    baseline_reasons,
    change_score,
    decide,
    ewma,
    median_and_spread,
    sigma_for_user,
)


def _sine(seconds: float, freq: float, amp: float = 0.2) -> np.ndarray:
    time = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * time)).astype(np.float32)


def test_decision_bands_and_recheck():
    low = decide(0.4, in_recheck=False, mild_streak=0, tau1=1.0, tau2=3.0, k_consecutive=2)
    assert low.state == "no_change" and low.mild_streak == 0
    mild = decide(2.0, in_recheck=False, mild_streak=0, tau1=1.0, tau2=3.0, k_consecutive=2)
    assert mild.state == "no_change" and mild.mild_streak == 1
    mild_again = decide(2.0, in_recheck=False, mild_streak=1, tau1=1.0, tau2=3.0, k_consecutive=2)
    assert mild_again.state == "recheck"
    high = decide(3.2, in_recheck=False, mild_streak=1, tau1=1.0, tau2=3.0, k_consecutive=2)
    assert high.state == "recheck" and high.mild_streak == 0
    caution = decide(3.2, in_recheck=True, mild_streak=0, tau1=1.0, tau2=3.0, k_consecutive=2)
    assert caution.state == "caution"


def test_score_ewma_and_sigma():
    distance_score, distance = change_score(np.array([1.0, 0.0]), np.array([0.0, 0.0]), 2.0)
    assert distance == pytest.approx(1.0)
    assert distance_score == pytest.approx(0.5)
    assert ewma(None, 2.0, 0.5) == 2.0
    assert ewma(2.0, 4.0, 0.5) == pytest.approx(3.0)
    assert sigma_for_user(None, 0, 0.08, 5) == 0.08
    assert sigma_for_user(0.2, 2, 0.08, 5) == 0.08
    assert sigma_for_user(0.2, 5, 0.08, 5) == pytest.approx(0.2)


def test_baseline_median_and_reasons():
    vectors = [np.array([1.0, 0.0]), np.array([1.1, 0.0]), np.array([0.9, 0.0])]
    median, spread = median_and_spread(vectors)
    assert median.shape == (2,)
    assert spread < 0.2
    assert baseline_reasons(1, 0.0, 2, 0.05) == ["BASELINE_TOO_FEW"]
    assert baseline_reasons(3, 0.2, 2, 0.05) == ["BASELINE_UNSTABLE"]
    assert baseline_reasons(3, 0.01, 2, 0.05) == []


def test_mel_pool_same_pitch_is_closer_and_registry_loads():
    extractor = MelPoolExtractor()
    base = extractor.embed(_sine(2.0, 200.0))
    near = extractor.embed(_sine(2.0, 200.0, amp=0.23))
    far = extractor.embed(_sine(2.0, 480.0))
    assert np.linalg.norm(base - near) < np.linalg.norm(base - far)
    registry = load_registry(Path(__file__).resolve().parents[2] / "registry" / "model_registry.yaml")
    assert registry.embedding_backend == "mel_pool"
    assert registry.tau1 < registry.tau2
    assert registry.extractor().name == "mel_pool"
    near_score, _ = change_score(near, base, registry.population_sigma)
    far_score, _ = change_score(far, base, registry.population_sigma)
    assert near_score < registry.tau1
    assert far_score >= registry.tau2


def test_acoustic_delta_is_finite():
    samples = _sine(2.0, 180.0)
    other = _sine(2.0, 220.0)
    delta = acoustic_delta(other, mfcc_mean(samples), cpp_mean(samples))
    assert np.isfinite(delta["mfcc_l2"])
    assert np.isfinite(delta["cpp_delta"])


def test_optional_models_and_classifier_stay_inactive():
    head = ClassifierHead()
    assert head.active is False
    with pytest.raises(RuntimeError):
        head.predict(np.zeros(4, dtype=np.float32))
    try:
        import models.mn.model as mn_model
    except ImportError:
        mn_model = None
    if mn_model is None:
        with pytest.raises(EmbeddingUnavailable):
            EfficientATExtractor("mn40_as", pretrained=False).embed(np.zeros(SAMPLE_RATE, dtype=np.float32))
    try:
        import transformers as transformers_mod
    except ImportError:
        transformers_mod = None
    if transformers_mod is None:
        with pytest.raises(EmbeddingUnavailable):
            WavLMExtractor("microsoft/wavlm-base-plus").embed(np.zeros(SAMPLE_RATE, dtype=np.float32))
