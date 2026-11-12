"""Replay complete local meals through the deployed TCN decision policy.

Reports swallow candidate event metrics, never aspiration/clinical accuracy.
Untrackable annotated swallows remain in the denominator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from swallow_ml.temporal import FEATURE_VERSION, TCN


def load_model(path: Path):
    model = json.loads(path.read_text(encoding="utf-8"))["model"]
    if not model or model.get("feature_version") != FEATURE_VERSION or model.get("sample_hz") != 8 or model.get("frames") != 16:
        raise ValueError("Expected a trained landmark-lk6-v1, 8 Hz, 16-frame model")
    if not model.get("provenance", {}).get("trained"):
        raise ValueError("Untrained weights cannot be evaluated as a deployed model")
    layers = model.get("layers")
    if not isinstance(layers, list) or len(layers) != 2:
        raise ValueError("Exactly two convolution layers are required")
    channels = 6
    for i, layer in enumerate(layers, 1):
        bias = np.asarray(layer["bias"], dtype=float)
        weights = np.asarray(layer["weights"], dtype=float)
        if layer["dilation"] != i or bias.ndim != 1 or not 1 <= bias.size <= 64 or weights.shape != (bias.size, channels, 3):
            raise ValueError("Invalid convolution shape or dilation")
        channels = bias.size
    if np.asarray(model["head"]["weights"]).shape != (channels,):
        raise ValueError("Invalid classifier head shape")
    network = TCN(hidden=len(model["layers"][0]["bias"]))
    for i, layer in enumerate(model["layers"], 1):
        if layer["dilation"] != i:
            raise ValueError("Unsupported dilation")
        network.parameters[f"w{i}"] = np.asarray(layer["weights"], dtype=float)
        network.parameters[f"b{i}"] = np.asarray(layer["bias"], dtype=float)
    network.parameters["head"] = np.asarray(model["head"]["weights"], dtype=float)
    network.parameters["bias"] = np.asarray([model["head"]["bias"]], dtype=float)
    mean, scale = np.asarray(model["mean"], dtype=float), np.asarray(model["scale"], dtype=float)
    if mean.shape != (6,) or scale.shape != (6,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid feature normalization")
    if not all(np.isfinite(value).all() for value in network.parameters.values()) or not 0 < model["threshold"] < 1:
        raise ValueError("Invalid model weights or threshold")
    return model, network


class ReplayWatch:
    """Equivalent to app/src/swallow/temporal.ts TemporalWatch."""
    def __init__(self, model, network):
        self.model, self.network = model, network
        self.history = []
        self.cooldown = 0

    def reset(self):
        self.history = []
        self.cooldown = 0

    def push(self, features, timestamp):
        values = np.asarray(features, dtype=float)
        if values.shape != (6,) or not np.isfinite(values).all() or not np.isfinite(timestamp):
            self.reset(); return None
        if self.history and (timestamp <= self.history[-1][0] or timestamp - self.history[-1][0] > 400):
            self.reset()
        self.history.append((timestamp, values))
        self.history = [row for row in self.history if timestamp - row[0] <= 2250]
        if timestamp < self.cooldown or timestamp - self.history[0][0] < 1875:
            return None
        window = []
        for i in range(16):
            target = timestamp - (15 - i) * 125
            right = next((j for j, row in enumerate(self.history) if row[0] >= target), None)
            if right is None:
                return None
            after = self.history[right]
            if after[0] == target:
                window.append(after[1]); continue
            if right == 0:
                return None
            before = self.history[right - 1]
            if after[0] - before[0] > 250:
                return None
            fraction = (target - before[0]) / (after[0] - before[0])
            window.append(before[1] + fraction * (after[1] - before[1]))
        inputs = (np.asarray(window) - self.model["mean"]) / self.model["scale"]
        probability = float(self.network.predict(inputs[None])[0])
        if probability < self.model["threshold"]:
            return None
        self.cooldown = timestamp + 2500
        self.history = []
        return probability


def event_metrics(annotations, predictions, duration_ms, early_ms=250, late_ms=1000):
    truth = np.asarray(annotations, dtype=float)
    detected = np.asarray(predictions, dtype=float)
    if truth.ndim != 1 or detected.ndim != 1 or not np.isfinite(truth).all() or not np.isfinite(detected).all():
        raise ValueError("Event times must be finite one-dimensional arrays")
    if not all(np.isfinite(value) for value in (duration_ms, early_ms, late_ms)) or duration_ms <= 0 or early_ms < 0 or late_ms < 0:
        raise ValueError("Positive duration and nonnegative matching tolerances required")
    if (truth < 0).any() or (truth > duration_ms).any() or (detected < 0).any() or (detected > duration_ms).any():
        raise ValueError("Events must be inside the measured video duration")
    truth = np.sort(truth)
    if len(set(truth.tolist())) != truth.size:
        raise ValueError("Duplicate annotations are not distinct swallows")
    used, delays = set(), []
    for timestamp in sorted(detected):
        match = next((i for i, value in enumerate(truth) if i not in used and value - early_ms <= timestamp <= value + late_ms), None)
        if match is not None:
            used.add(match); delays.append(float(timestamp - truth[match]))
    tp, fp, fn = len(used), detected.size - len(used), truth.size - len(used)
    return {"tp": tp, "fp": int(fp), "fn": int(fn), "duration_ms": float(duration_ms),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "false_alarms_per_hour": float(fp / (duration_ms / 3_600_000)),
            "detection_delays_ms": delays}


def evaluate(manifest: Path, model_path: Path, face_model: Path, early_ms=250, late_ms=1000):
    import cv2
    from swallow_ml.motion import MotionExtractor
    model, network = load_model(model_path)
    split = model["provenance"].get("split_subjects")
    if not split or not {"train", "validation", "test"} <= split.keys():
        raise ValueError("Subject split provenance is required for held-out evaluation")
    excluded = set(split["train"]) | set(split["validation"])
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Expected a nonempty annotated video manifest")
    resolved = [(str(entry["subject_id"]), (manifest.parent / entry["video"]).resolve()) for entry in entries]
    if any(subject in excluded for subject, _ in resolved):
        raise ValueError("Evaluation includes a training or validation subject")
    if len({path for _, path in resolved}) != len(resolved):
        raise ValueError("A video may only appear once in an evaluation")
    extractor = MotionExtractor(face_model)
    reports = []
    try:
        for entry, (subject, path) in zip(entries, resolved):
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise ValueError(f"Cannot open video: {path.name}")
            try:
                fps = capture.get(cv2.CAP_PROP_FPS)
                if not np.isfinite(fps) or fps < 8:
                    raise ValueError("A known input frame rate of at least 8 fps is required")
                extractor.reset(); watch = ReplayWatch(model, network)
                frame_index = 0; next_time = 0; attempts = 0; tracked = 0; detections = []
                while True:
                    ok, frame = capture.read()
                    if not ok: break
                    timestamp = frame_index * 1000 / fps; frame_index += 1
                    if timestamp + .001 < next_time: continue
                    next_time += 125; attempts += 1
                    features = extractor.push(frame, timestamp)
                    if features is None:
                        watch.reset(); continue
                    tracked += 1
                    if watch.push(features, timestamp) is not None:
                        detections.append(timestamp)
                metrics = event_metrics(entry["swallows_ms"], detections, frame_index * 1000 / fps, early_ms, late_ms)
                reports.append({"subject_id": subject, "source": entry["video"], "detections_ms": detections,
                                "tracking_samples": tracked, "attempted_samples": attempts, **metrics})
            finally:
                capture.release()
    finally:
        extractor.close()
    totals = {key: sum(row[key] for row in reports) for key in ("tp", "fp", "fn", "duration_ms", "tracking_samples", "attempted_samples")}
    tp, fp, fn = (totals[key] for key in ("tp", "fp", "fn"))
    return {"scope": "held_out_swallow_candidate_events", "clinical_validation": False,
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "matching": {"early_ms": early_ms, "late_ms": late_ms, "cooldown_ms": 2500},
            "totals": {**totals, "precision": tp / (tp + fp) if tp + fp else None,
                       "recall": tp / (tp + fn) if tp + fn else None,
                       "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                       "false_alarms_per_hour": fp / (totals["duration_ms"] / 3_600_000)},
            "videos": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--face-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--early-ms", type=float, default=250)
    parser.add_argument("--late-ms", type=float, default=1000)
    args = parser.parse_args()
    report = evaluate(args.manifest, args.model, args.face_model, args.early_ms, args.late_ms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["totals"], indent=2))


if __name__ == "__main__":
    main()
