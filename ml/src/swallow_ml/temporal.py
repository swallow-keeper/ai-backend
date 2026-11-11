"""Small trainable causal TCN and a portable JSON inference format.

NumPy training avoids a framework-specific mobile runtime. Weights are exported
without changing operations; app/src/swallow/temporal.ts executes the same graph.
This classifier detects annotated swallow candidates, never aspiration.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

FEATURE_VERSION = "landmark-lk6-v1"


class TCN:
    def __init__(self, seed: int = 42, hidden: int = 12):
        rng = np.random.default_rng(seed)
        self.parameters = {
            "w1": rng.normal(0, np.sqrt(2 / 18), (hidden, 6, 3)),
            "b1": np.zeros(hidden),
            "w2": rng.normal(0, np.sqrt(2 / (hidden * 3)), (hidden, hidden, 3)),
            "b2": np.zeros(hidden),
            "head": rng.normal(0, np.sqrt(1 / hidden), hidden),
            "bias": np.zeros(1),
        }

    @staticmethod
    def conv(x, weights, bias, dilation):
        out = np.broadcast_to(bias, (*x.shape[:2], bias.size)).copy()
        for tap in range(3):
            lag = (2 - tap) * dilation
            out[:, lag:] += x[:, :x.shape[1] - lag] @ weights[:, :, tap].T
        return out

    def forward(self, x):
        p = self.parameters
        z1 = self.conv(x, p["w1"], p["b1"], 1)
        a1 = np.maximum(z1, 0)
        z2 = self.conv(a1, p["w2"], p["b2"], 2)
        pooled = np.maximum(z2, 0).mean(axis=1)
        logits = pooled @ p["head"] + p["bias"][0]
        return logits, (x, z1, a1, z2, pooled)

    def predict(self, x):
        logits, _ = self.forward(x)
        return 1 / (1 + np.exp(-np.clip(logits, -80, 80)))

    @staticmethod
    def conv_backward(x, weights, derivative, dilation):
        dx = np.zeros_like(x)
        dw = np.zeros_like(weights)
        for tap in range(3):
            lag = (2 - tap) * dilation
            source = x[:, :x.shape[1] - lag]
            target = derivative[:, lag:]
            dw[:, :, tap] = np.einsum("bto,bti->oi", target, source)
            dx[:, :x.shape[1] - lag] += target @ weights[:, :, tap]
        return dx, dw, derivative.sum(axis=(0, 1))

    def loss_and_gradients(self, x, labels):
        logits, (x, z1, a1, z2, pooled) = self.forward(x)
        loss = np.mean(np.logaddexp(0, logits) - labels * logits)
        derivative = (1 / (1 + np.exp(-np.clip(logits, -80, 80))) - labels) / labels.size
        p = self.parameters
        grads = {"head": pooled.T @ derivative, "bias": np.array([derivative.sum()])}
        da2 = np.broadcast_to((derivative[:, None] * p["head"])[:, None, :] / x.shape[1], z2.shape)
        da1, grads["w2"], grads["b2"] = self.conv_backward(a1, p["w2"], da2 * (z2 > 0), 2)
        _, grads["w1"], grads["b1"] = self.conv_backward(x, p["w1"], da1 * (z1 > 0), 1)
        return float(loss), grads

    def export(self, mean, scale, threshold, provenance):
        p = self.parameters
        return {
            "version": "swallow-tcn-v1", "feature_version": FEATURE_VERSION,
            "sample_hz": 8, "frames": 16, "threshold": float(threshold),
            "mean": mean.tolist(), "scale": scale.tolist(),
            "layers": [
                {"dilation": 1, "weights": p["w1"].tolist(), "bias": p["b1"].tolist()},
                {"dilation": 2, "weights": p["w2"].tolist(), "bias": p["b2"].tolist()},
            ],
            "head": {"weights": p["head"].tolist(), "bias": float(p["bias"][0])},
            "provenance": provenance,
        }


def metrics(labels, probabilities, threshold):
    predicted = probabilities >= threshold
    positive = labels == 1
    tp = int(np.sum(predicted & positive)); fp = int(np.sum(predicted & ~positive))
    fn = int(np.sum(~predicted & positive)); tn = int(np.sum(~predicted & ~positive))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "specificity": tn / max(1, tn + fp), "f1": 2 * tp / max(1, 2 * tp + fp + fn)}


def read_dataset(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Dataset is empty")
    for row in rows:
        if row.get("feature_version") != FEATURE_VERSION or row.get("label") not in (0, 1) or not row.get("subject_id"):
            raise ValueError("Each example needs feature_version, subject_id, features, and label 0 or 1")
    x = np.asarray([row["features"] for row in rows], dtype=np.float64)
    if x.shape != (len(rows), 16, 6) or not np.isfinite(x).all():
        raise ValueError("Expected finite motion features of shape [examples, 16, 6]")
    return rows, x, np.asarray([row["label"] for row in rows], dtype=np.float64)


def split_subjects(rows, seed):
    subjects = sorted({str(row["subject_id"]) for row in rows})
    if len(subjects) < 6:
        raise ValueError("At least six subjects are required for separate train, validation, and test groups")
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    held = max(1, len(subjects) // 5)
    groups = [set(subjects[:-2 * held]), set(subjects[-2 * held:-held]), set(subjects[-held:])]
    indices = [np.array([i for i, row in enumerate(rows) if str(row["subject_id"]) in group]) for group in groups]
    for index in indices:
        if {rows[int(i)]["label"] for i in index} != {0, 1}:
            raise ValueError("Every subject split must contain positive and negative examples")
    return indices, groups


def train(path: Path, epochs=100, seed=42):
    rows, x, labels = read_dataset(path)
    (training, validation, testing), groups = split_subjects(rows, seed)
    mean = x[training].mean(axis=(0, 1)); scale = np.maximum(x[training].std(axis=(0, 1)), 1e-4)
    x = (x - mean) / scale
    model = TCN(seed)
    rng = np.random.default_rng(seed)
    first = {key: np.zeros_like(value) for key, value in model.parameters.items()}
    second = copy.deepcopy(first)
    best = None; best_loss = float("inf"); step = 0
    for _epoch in range(epochs):
        order = rng.permutation(training)
        for offset in range(0, order.size, 64):
            batch = order[offset:offset + 64]
            _, grads = model.loss_and_gradients(x[batch], labels[batch])
            step += 1
            for key in model.parameters:
                grad = np.clip(grads[key], -5, 5)
                first[key] = 0.9 * first[key] + 0.1 * grad
                second[key] = 0.999 * second[key] + 0.001 * grad ** 2
                model.parameters[key] -= 0.001 * (first[key] / (1 - 0.9 ** step)) / (np.sqrt(second[key] / (1 - 0.999 ** step)) + 1e-8)
        loss, _ = model.loss_and_gradients(x[validation], labels[validation])
        if loss < best_loss:
            best_loss = loss; best = copy.deepcopy(model.parameters)
    if best is None:
        raise ValueError("epochs must be positive")
    model.parameters = best
    probabilities = model.predict(x[validation])
    thresholds = np.linspace(0.05, 0.95, 91)
    threshold = max(thresholds, key=lambda value: metrics(labels[validation], probabilities, value)["f1"])
    provenance = {
        "trained": True, "subjects": len(set.union(*groups)),
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "validation": metrics(labels[validation], probabilities, threshold),
        "test": metrics(labels[testing], model.predict(x[testing]), threshold),
        "split_subjects": {name: sorted(group) for name, group in zip(("train", "validation", "test"), groups)},
        "seed": seed, "epochs": epochs, "clinical_validation": False,
    }
    return model.export(mean, scale, threshold, provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    model = train(args.dataset, args.epochs, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status": "trained_research_model", "model": model}, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(model["provenance"], indent=2))


if __name__ == "__main__":
    main()
