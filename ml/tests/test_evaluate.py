import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from swallow_ml.evaluate import ReplayWatch, evaluate, event_metrics, load_model
from swallow_ml.temporal import TCN


def test_one_to_one_matching_counts_duplicates_and_untrackable_misses():
    result = event_metrics([1000, 3000, 5000], [900, 1100, 3500, 7000], 10000)
    assert (result["tp"], result["fp"], result["fn"]) == (2, 2, 1)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["false_alarms_per_hour"] == pytest.approx(720)
    assert result["detection_delays_ms"] == [-100, 500]


def test_empty_cases_are_not_claimed_as_perfect_performance():
    result = event_metrics([], [], 1000)
    assert result["precision"] is None and result["recall"] is None
    assert event_metrics([500], [], 1000)["recall"] == 0
    with pytest.raises(ValueError, match="duration"):
        event_metrics([], [], 0)
    with pytest.raises(ValueError, match="inside"):
        event_metrics([2000], [], 1000)
    with pytest.raises(ValueError, match="Duplicate"):
        event_metrics([500, 500], [], 1000)


def test_replay_uses_app_window_and_cooldown():
    network = TCN(hidden=3)
    network.parameters["head"][:] = 0
    network.parameters["bias"][:] = 10
    model = network.export(np.zeros(6), np.ones(6), .7, {"trained": False})
    watch = ReplayWatch(model, network)
    for i in range(15):
        assert watch.push([0] * 6, i * 125) is None
    assert watch.push([0] * 6, 1875) > .99
    for i in range(16, 35):
        assert watch.push([0] * 6, i * 125) is None
    assert watch.push([0] * 6, 4375) > .99
    assert watch.push([0] * 6, 10000) is None


def test_loading_refuses_untrained_weights(tmp_path):
    network = TCN(hidden=3)
    path = tmp_path / "test-only.json"
    model = network.export(np.zeros(6), np.ones(6), .7, {"trained": False})
    path.write_text(json.dumps({"model": model}))
    with pytest.raises(ValueError, match="Untrained"):
        load_model(path)
    model["provenance"]["trained"] = True  # Test only, never deploy this fixture.
    path.write_text(json.dumps({"model": model}))
    _, restored = load_model(path)
    window = np.ones((1, 16, 6))
    assert restored.predict(window) == pytest.approx(network.predict(window))
    model["layers"][0]["weights"][0].pop()
    path.write_text(json.dumps({"model": model}))
    with pytest.raises(ValueError, match="shape"):
        load_model(path)


def test_full_replay_retains_untrackable_truth_and_rejects_subject_leakage(tmp_path, monkeypatch):
    import swallow_ml.motion as motion

    class Capture:
        def __init__(self, path): self.frames = 0; self.released = False
        def isOpened(self): return True
        def get(self, key): return 25
        def read(self):
            self.frames += 1
            return (True, None) if self.frames <= 100 else (False, None)
        def release(self): self.released = True

    class Extractor:
        visible = False
        def __init__(self, path): pass
        def reset(self): pass
        def close(self): pass
        def push(self, frame, timestamp): return [0] * 6 if self.visible else None

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(VideoCapture=Capture, CAP_PROP_FPS=5))
    monkeypatch.setattr(motion, "MotionExtractor", Extractor)
    network = TCN(hidden=3)
    network.parameters["head"][:] = 0; network.parameters["bias"][:] = 10
    model = network.export(np.zeros(6), np.ones(6), .7, {
        "trained": True, "split_subjects": {"train": ["training"], "validation": ["validation"], "test": ["heldout"]}})
    weights = tmp_path / "test-only-model.json"
    weights.write_text(json.dumps({"model": model}))
    manifest = tmp_path / "test-manifest.json"
    entry = {"subject_id": "heldout", "video": "mock-video.avi", "swallows_ms": [1880, 3000]}
    manifest.write_text(json.dumps([entry]))
    report = evaluate(manifest, weights, tmp_path / "face.task")
    assert report["totals"]["fn"] == 2
    assert report["totals"]["tracking_samples"] == 0
    assert report["totals"]["recall"] == 0
    assert report["clinical_validation"] is False
    Extractor.visible = True
    report = evaluate(manifest, weights, tmp_path / "face.task")
    assert report["totals"]["tp"] == 1 and report["totals"]["fn"] == 1
    assert report["videos"][0]["detections_ms"] == [1880]
    entry["subject_id"] = "training"
    manifest.write_text(json.dumps([entry]))
    with pytest.raises(ValueError, match="training or validation"):
        evaluate(manifest, weights, tmp_path / "face.task")


def test_replay_interpolates_jitter_but_does_not_bridge_missing_frames():
    network = TCN(hidden=3)
    model = network.export(np.zeros(6), np.ones(6), 1e-10, {"trained": False})
    watch = ReplayWatch(model, network)
    for i in range(15):
        assert watch.push([i * .13] * 6, i * 130) is None
    expected = np.array([[(1950 - (15 - i) * 125) / 1000] * 6 for i in range(16)])
    assert watch.push([1.95] * 6, 1950) == pytest.approx(network.predict(expected[None])[0])
    missing = ReplayWatch(model, network)
    for i in range(16):
        if i not in (5, 6):
            assert missing.push([i * .13] * 6, i * 130) is None
