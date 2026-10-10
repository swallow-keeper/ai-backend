#!/usr/bin/env python3
# Reconstruct swallow events from window scores and evaluate them against VFSS labels.

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from audit_dataset import frame_to_sample, read_label_rows
from train_baseline import ModelConfig, SwallowCNN, SwallowWindowDataset, choose_device


@dataclass(frozen=True)
class Event:
    start_sample: int
    end_sample: int
    score: float | None = None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate swallow events on the held-out test split.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed/dataset"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/baseline/best_model.pt"))
    parser.add_argument("--config", type=Path, default=Path("configs/event_evaluation.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/evaluation"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser.parse_args()


def load_rows(dataset_dir: Path) -> list[dict[str, str]]:
    with (dataset_dir / "windows.csv").open(newline="", encoding="utf-8") as input_file:
        return [row for row in csv.DictReader(input_file) if row["split"] == "test"]


def interval_iou(left: Event, right: Event) -> float:
    intersection = max(0, min(left.end_sample, right.end_sample) - max(left.start_sample, right.start_sample))
    union = max(left.end_sample, right.end_sample) - min(left.start_sample, right.start_sample)
    return intersection / union if union else 0.0


def scores_to_events(
    signal_samples: int,
    window_scores: list[tuple[int, int, float]],
    score_threshold: float,
    minimum_event_samples: int,
    merge_gap_samples: int,
) -> list[Event]:
    sums = np.zeros(signal_samples, dtype=np.float32)
    counts = np.zeros(signal_samples, dtype=np.uint16)
    for start, end, score in window_scores:
        sums[start:end] += score
        counts[start:end] += 1

    scores = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    active = scores >= score_threshold
    change_indices = np.flatnonzero(np.diff(active.astype(np.int8))) + 1
    boundaries = np.concatenate(([0], change_indices, [signal_samples]))
    raw_events = [
        Event(int(start), int(end), float(scores[start:end].mean()))
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
        if active[start]
    ]

    merged: list[Event] = []
    for event in raw_events:
        if merged and event.start_sample - merged[-1].end_sample <= merge_gap_samples:
            previous = merged.pop()
            merged.append(Event(previous.start_sample, event.end_sample, None))
        else:
            merged.append(event)

    return [
        Event(event.start_sample, event.end_sample, float(scores[event.start_sample:event.end_sample].mean()))
        for event in merged
        if event.end_sample - event.start_sample >= minimum_event_samples
    ]


def match_events(
    predictions: list[Event],
    targets: list[Event],
    minimum_iou: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    candidates = sorted(
        (
            (prediction_index, target_index, interval_iou(prediction, target))
            for prediction_index, prediction in enumerate(predictions)
            for target_index, target in enumerate(targets)
            if interval_iou(prediction, target) >= minimum_iou
        ),
        key=lambda candidate: candidate[2],
        reverse=True,
    )
    matched_predictions: set[int] = set()
    matched_targets: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for prediction_index, target_index, iou in candidates:
        if prediction_index in matched_predictions or target_index in matched_targets:
            continue
        matches.append((prediction_index, target_index, iou))
        matched_predictions.add(prediction_index)
        matched_targets.add(target_index)
    return (
        matches,
        [index for index in range(len(predictions)) if index not in matched_predictions],
        [index for index in range(len(targets)) if index not in matched_targets],
    )


def event_metrics(
    predictions_by_file: dict[str, list[Event]],
    targets_by_file: dict[str, list[Event]],
    total_duration_seconds: float,
    minimum_iou: float,
    sample_rate_hz: int,
) -> tuple[dict[str, float], dict[str, object]]:
    matches_by_file: dict[str, list[dict[str, float]]] = {}
    false_positive_count = 0
    false_negative_count = 0
    true_positive_count = 0
    onset_errors: list[float] = []
    offset_errors: list[float] = []
    ious: list[float] = []

    for file_name in sorted(targets_by_file):
        predictions = predictions_by_file.get(file_name, [])
        targets = targets_by_file[file_name]
        matches, false_positives, false_negatives = match_events(predictions, targets, minimum_iou)
        true_positive_count += len(matches)
        false_positive_count += len(false_positives)
        false_negative_count += len(false_negatives)
        matches_by_file[file_name] = []
        for prediction_index, target_index, iou in matches:
            prediction = predictions[prediction_index]
            target = targets[target_index]
            onset_error_seconds = (prediction.start_sample - target.start_sample) / sample_rate_hz
            offset_error_seconds = (prediction.end_sample - target.end_sample) / sample_rate_hz
            onset_errors.append(onset_error_seconds)
            offset_errors.append(offset_error_seconds)
            ious.append(iou)
            matches_by_file[file_name].append(
                {
                    "prediction_index": prediction_index,
                    "target_index": target_index,
                    "iou": iou,
                    "onset_error_seconds": onset_error_seconds,
                    "offset_error_seconds": offset_error_seconds,
                }
            )

    precision = true_positive_count / (true_positive_count + false_positive_count) if true_positive_count + false_positive_count else 0.0
    recall = true_positive_count / (true_positive_count + false_negative_count) if true_positive_count + false_negative_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "false_positives_per_minute": false_positive_count / (total_duration_seconds / 60),
        "mean_iou_for_matches": float(np.mean(ious)) if ious else 0.0,
        "mean_onset_error_seconds": float(np.mean(onset_errors)) if onset_errors else 0.0,
        "mean_offset_error_seconds": float(np.mean(offset_errors)) if offset_errors else 0.0,
        "true_positives": true_positive_count,
        "false_positives": false_positive_count,
        "false_negatives": false_negative_count,
    }
    return metrics, {"matches_by_file": matches_by_file}


def load_targets(
    raw_dir: Path,
    file_names: set[str],
    sample_rate_hz: int,
    frame_rate_hz: int,
) -> tuple[dict[str, list[Event]], dict[str, int]]:
    targets_by_file: dict[str, list[Event]] = {}
    signal_lengths: dict[str, int] = {}
    for file_name in sorted(file_names):
        signal = np.load(raw_dir / "signals" / f"{file_name}.npy", mmap_mode="r")
        signal_lengths[file_name] = signal.shape[0]
        targets = []
        for row in read_label_rows(raw_dir / "labels" / f"{file_name}.csv"):
            start = frame_to_sample(float(row["start"]), sample_rate_hz, frame_rate_hz)
            end = frame_to_sample(float(row["end"]), sample_rate_hz, frame_rate_hz)
            if 0 <= start < end <= signal.shape[0]:
                targets.append(Event(start, end))
        targets_by_file[file_name] = targets
    return targets_by_file, signal_lengths


def predict_window_scores(
    model: SwallowCNN,
    rows: list[dict[str, str]],
    raw_dir: Path,
    normalizer: dict[str, object],
    window_samples: int,
    device: torch.device,
) -> dict[str, list[tuple[int, int, float]]]:
    dataset = SwallowWindowDataset(rows, raw_dir, normalizer, window_samples)
    loader = DataLoader(dataset, batch_size=64)
    scores_by_file: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    model.eval()
    row_offset = 0
    with torch.no_grad():
        for values, _ in loader:
            probabilities = torch.sigmoid(model(values.to(device))).cpu().tolist()
            batch_rows = rows[row_offset : row_offset + len(probabilities)]
            for row, probability in zip(batch_rows, probabilities, strict=True):
                scores_by_file[row["file_name"]].append(
                    (int(row["window_start"]), int(row["window_end"]), probability)
                )
            row_offset += len(probabilities)
    return dict(scores_by_file)


def main() -> None:
    arguments = parse_arguments()
    evaluation_config = json.loads(arguments.config.read_text(encoding="utf-8"))
    dataset_summary = json.loads((arguments.dataset_dir / "summary.json").read_text(encoding="utf-8"))
    normalizer = json.loads((arguments.dataset_dir / "normalizer.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    model_config = ModelConfig(
        input_channels=checkpoint["model_config"]["input_channels"],
        channels=tuple(checkpoint["model_config"]["channels"]),
        dropout=checkpoint["model_config"]["dropout"],
    )
    model = SwallowCNN(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = choose_device(arguments.device)
    model.to(device)
    rows = load_rows(arguments.dataset_dir)
    sample_rate_hz = int(dataset_summary["config"]["sample_rate_hz"])
    frame_rate_hz = int(dataset_summary["config"]["vfss_frame_rate_hz"])
    window_samples = int(dataset_summary["config"]["window_samples"])
    scores_by_file = predict_window_scores(
        model,
        rows,
        arguments.raw_dir,
        normalizer,
        window_samples,
        device,
    )
    targets_by_file, signal_lengths = load_targets(
        arguments.raw_dir,
        set(scores_by_file),
        sample_rate_hz,
        frame_rate_hz,
    )
    predictions_by_file = {
        file_name: scores_to_events(
            signal_lengths[file_name],
            scores,
            evaluation_config["score_threshold"],
            round(evaluation_config["minimum_event_seconds"] * sample_rate_hz),
            round(evaluation_config["merge_gap_seconds"] * sample_rate_hz),
        )
        for file_name, scores in scores_by_file.items()
    }
    metrics, match_details = event_metrics(
        predictions_by_file,
        targets_by_file,
        sum(signal_lengths.values()) / sample_rate_hz,
        evaluation_config["minimum_iou"],
        sample_rate_hz,
    )
    output = {
        "evaluation_config": evaluation_config,
        "metrics": metrics,
        "predictions_by_file": {
            file_name: [event.__dict__ for event in events]
            for file_name, events in predictions_by_file.items()
        },
        **match_details,
    }
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.output_dir / "event_report.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
