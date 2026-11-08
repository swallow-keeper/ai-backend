import json

import numpy as np
import pytest

from swallow_ml.temporal import TCN, FEATURE_VERSION, read_dataset, split_subjects, train


def test_tcn_gradients_match_finite_difference():
    model = TCN(hidden=3)
    rng = np.random.default_rng(2)
    # Move away from ReLU's nondifferentiable all-zero initial bias.
    model.parameters["b1"][:] = rng.normal(0.1, 0.03, 3)
    model.parameters["b2"][:] = rng.normal(0.1, 0.03, 3)
    x = rng.normal(size=(3, 16, 6)); labels = np.array([0., 1., 0.])
    _, gradients = model.loss_and_gradients(x, labels)
    for key, parameter in model.parameters.items():
        index = tuple(0 for _ in parameter.shape)
        original = parameter[index]
        parameter[index] = original + 1e-5
        plus, _ = model.loss_and_gradients(x, labels)
        parameter[index] = original - 1e-5
        minus, _ = model.loss_and_gradients(x, labels)
        parameter[index] = original
        assert gradients[key][index] == pytest.approx((plus - minus) / 2e-5, abs=1e-6), key


def test_subject_splits_do_not_leak_and_training_reduces_loss(tmp_path):
    # Synthetic examples verify optimization only. They are never deployment data.
    rng = np.random.default_rng(4)
    path = tmp_path / "synthetic-unit-test.jsonl"
    rows = []
    for subject in range(6):
        for label in (0, 1):
            for _ in range(8):
                rows.append({"subject_id": str(subject), "label": label, "feature_version": FEATURE_VERSION,
                             "features": (rng.normal(0, .1, (16, 6)) + label).tolist()})
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    parsed, _, _ = read_dataset(path)
    _, groups = split_subjects(parsed, 42)
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    model = train(path, epochs=30)
    assert model["provenance"]["test"]["f1"] > .9
    assert model["provenance"]["clinical_validation"] is False
    assert model["feature_version"] == FEATURE_VERSION


def test_refuses_missing_data_and_insufficient_subjects(tmp_path):
    path = tmp_path / "empty.jsonl"; path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        read_dataset(path)
    with pytest.raises(ValueError, match="six subjects"):
        split_subjects([{"subject_id": "one"}], 42)
