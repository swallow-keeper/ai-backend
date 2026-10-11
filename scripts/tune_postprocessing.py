#!/usr/bin/env python3
# Select event post-processing settings on validation data only.

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import TypedDict

from evaluate_events import event_metrics, load_rows, load_targets, predict_window_scores, scores_to_events
from model_artifacts import file_sha256, load_checkpoint
from train_baseline import choose_device


class EventConfig(TypedDict):
    score_threshold: float
    minimum_event_seconds: float
    merge_gap_seconds: float
    minimum_iou: float


class TuningCandidate(TypedDict):
    event_config: EventConfig
    metrics: dict[str, float]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune event reconstruction settings on validation data.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed/dataset"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/postprocess_tuning.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    search = json.loads(arguments.config.read_text(encoding="utf-8"))
    summary = json.loads((arguments.dataset_dir / "summary.json").read_text(encoding="utf-8"))
    device = choose_device(arguments.device)
    model, checkpoint = load_checkpoint(arguments.checkpoint, device)
    if checkpoint["dataset"]["splits_sha256"] != file_sha256(arguments.dataset_dir / "splits.json"):
        raise ValueError("Checkpoint and dataset split manifests do not match.")

    rows = load_rows(arguments.dataset_dir, "validation")
    sample_rate_hz = int(summary["config"]["sample_rate_hz"])
    frame_rate_hz = int(summary["config"]["vfss_frame_rate_hz"])
    window_samples = int(summary["config"]["window_samples"])
    scores_by_file = predict_window_scores(
        model,
        rows,
        arguments.raw_dir,
        checkpoint["normalizer"],
        window_samples,
        device,
    )
    targets_by_file, signal_lengths = load_targets(
        arguments.raw_dir,
        set(scores_by_file),
        sample_rate_hz,
        frame_rate_hz,
    )
    total_duration_seconds = sum(signal_lengths.values()) / sample_rate_hz
    candidates: list[TuningCandidate] = []

    for threshold, minimum_seconds, merge_seconds in itertools.product(
        search["score_thresholds"],
        search["minimum_event_seconds"],
        search["merge_gap_seconds"],
    ):
        event_config: EventConfig = {
            "score_threshold": threshold,
            "minimum_event_seconds": minimum_seconds,
            "merge_gap_seconds": merge_seconds,
            "minimum_iou": search["minimum_iou"],
        }
        predictions_by_file = {
            file_name: scores_to_events(
                signal_lengths[file_name],
                scores,
                threshold,
                round(minimum_seconds * sample_rate_hz),
                round(merge_seconds * sample_rate_hz),
            )
            for file_name, scores in scores_by_file.items()
        }
        metrics, _ = event_metrics(
            predictions_by_file,
            targets_by_file,
            total_duration_seconds,
            search["minimum_iou"],
            sample_rate_hz,
        )
        candidates.append(TuningCandidate(event_config=event_config, metrics=metrics))

    selected = max(
        candidates,
        key=lambda candidate: (
            candidate["metrics"]["event_f1"],
            candidate["metrics"]["event_recall"],
            -candidate["metrics"]["false_positives_per_minute"],
        ),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(selected["event_config"], indent=2) + "\n",
        encoding="utf-8",
    )
    arguments.output.with_suffix(".tuning_report.json").write_text(
        json.dumps(
            {
                "selection_split": "validation",
                "selection_rule": "highest event F1, then recall, then lower false positives per minute",
                "checkpoint": str(arguments.checkpoint),
                "checkpoint_dataset_splits_sha256": checkpoint["dataset"]["splits_sha256"],
                "selected": selected,
                "candidates": candidates,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(selected, sort_keys=True))


if __name__ == "__main__":
    main()
