#!/usr/bin/env python3
# Build a reproducible, patient-grouped manifest for swallow-detector training.

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audit_dataset import FILENAME_PATTERN, frame_to_sample, read_label_rows


SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class DatasetConfig:
    sample_rate_hz: int
    vfss_frame_rate_hz: int
    window_seconds: float
    stride_seconds: float
    positive_overlap_ratio: float
    split_seed: int
    split_ratios: dict[str, float]

    @property
    def window_samples(self) -> int:
        return round(self.window_seconds * self.sample_rate_hz)

    @property
    def stride_samples(self) -> int:
        return round(self.stride_seconds * self.sample_rate_hz)


@dataclass(frozen=True)
class Recording:
    file_name: str
    participant: str
    signal_samples: int
    channels: int
    intervals: tuple[tuple[int, int], ...]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create patient-grouped data splits and window-level swallow labels."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--config", type=Path, default=Path("configs/dataset.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/dataset"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing generated dataset manifest.",
    )
    return parser.parse_args()


def load_config(path: Path) -> DatasetConfig:
    values = json.loads(path.read_text(encoding="utf-8"))
    config = DatasetConfig(**values)
    if set(config.split_ratios) != set(SPLIT_NAMES):
        raise ValueError(f"split_ratios must contain exactly {SPLIT_NAMES}.")
    if not np.isclose(sum(config.split_ratios.values()), 1.0):
        raise ValueError("split_ratios must sum to 1.0.")
    if config.window_samples <= 0 or config.stride_samples <= 0:
        raise ValueError("window_seconds and stride_seconds must produce positive sample counts.")
    if not 0 < config.positive_overlap_ratio <= 1:
        raise ValueError("positive_overlap_ratio must be in (0, 1].")
    return config


def load_recordings(raw_dir: Path, config: DatasetConfig) -> list[Recording]:
    signals_dir = raw_dir / "signals"
    labels_dir = raw_dir / "labels"
    recordings: list[Recording] = []

    for signal_path in sorted(signals_dir.glob("*.npy")):
        match = FILENAME_PATTERN.fullmatch(signal_path.stem)
        if match is None:
            raise ValueError(f"Unexpected filename format: {signal_path.name}")

        label_rows = read_label_rows(labels_dir / f"{signal_path.stem}.csv")
        if not label_rows:
            continue

        signal = np.load(signal_path, mmap_mode="r")
        intervals: list[tuple[int, int]] = []
        for row in label_rows:
            start = frame_to_sample(
                float(row["start"]),
                config.sample_rate_hz,
                config.vfss_frame_rate_hz,
            )
            end = frame_to_sample(
                float(row["end"]),
                config.sample_rate_hz,
                config.vfss_frame_rate_hz,
            )
            if 0 <= start < end <= signal.shape[0]:
                intervals.append((start, end))

        if not intervals:
            continue
        recordings.append(
            Recording(
                file_name=signal_path.stem,
                participant=match.group("participant").upper(),
                signal_samples=signal.shape[0],
                channels=signal.shape[1],
                intervals=tuple(intervals),
            )
        )
    return recordings


def participant_cohort(participant: str) -> str:
    return "NS" if participant.startswith("NS") else "S"


def split_participants(
    participants: set[str],
    split_ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for participant in participants:
        groups[participant_cohort(participant)].append(participant)

    assignments: dict[str, str] = {}
    for cohort, cohort_participants in sorted(groups.items()):
        random_generator = np.random.default_rng(seed + sum(ord(character) for character in cohort))
        ordered = sorted(cohort_participants)
        random_generator.shuffle(ordered)
        counts = {
            "train": round(len(ordered) * split_ratios["train"]),
            "validation": round(len(ordered) * split_ratios["validation"]),
        }
        counts["test"] = len(ordered) - counts["train"] - counts["validation"]
        if any(count <= 0 for count in counts.values()):
            raise ValueError(f"Cohort {cohort} is too small for all requested splits.")

        offset = 0
        for split_name in SPLIT_NAMES:
            for participant in ordered[offset : offset + counts[split_name]]:
                assignments[participant] = split_name
            offset += counts[split_name]
    return assignments


def window_starts(signal_samples: int, window_samples: int, stride_samples: int) -> list[int]:
    if signal_samples <= window_samples:
        return [0]
    starts = list(range(0, signal_samples - window_samples + 1, stride_samples))
    final_start = signal_samples - window_samples
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def overlap_samples(
    window_start: int,
    window_end: int,
    intervals: tuple[tuple[int, int], ...],
) -> int:
    return sum(
        max(0, min(window_end, interval_end) - max(window_start, interval_start))
        for interval_start, interval_end in intervals
    )


def build_window_rows(
    recordings: list[Recording],
    assignments: dict[str, str],
    config: DatasetConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    required_overlap = round(config.window_samples * config.positive_overlap_ratio)

    for recording in recordings:
        split = assignments[recording.participant]
        for start in window_starts(
            recording.signal_samples,
            config.window_samples,
            config.stride_samples,
        ):
            end = min(start + config.window_samples, recording.signal_samples)
            positive_samples = overlap_samples(start, end, recording.intervals)
            rows.append(
                {
                    "split": split,
                    "participant": recording.participant,
                    "file_name": recording.file_name,
                    "window_start": start,
                    "window_end": end,
                    "positive_samples": positive_samples,
                    "label": int(positive_samples >= required_overlap),
                }
            )
    return rows


def calculate_train_normalizer(
    raw_dir: Path,
    recordings: list[Recording],
    assignments: dict[str, str],
) -> dict[str, object]:
    sum_values: np.ndarray | None = None
    sum_squared_values: np.ndarray | None = None
    sample_count = 0

    for recording in recordings:
        if assignments[recording.participant] != "train":
            continue
        signal = np.load(raw_dir / "signals" / f"{recording.file_name}.npy", mmap_mode="r")
        values = np.asarray(signal, dtype=np.float64)
        if sum_values is None:
            sum_values = np.zeros(values.shape[1], dtype=np.float64)
            sum_squared_values = np.zeros(values.shape[1], dtype=np.float64)
        sum_values += values.sum(axis=0)
        sum_squared_values += np.square(values).sum(axis=0)
        sample_count += values.shape[0]

    if sum_values is None or sum_squared_values is None or sample_count == 0:
        raise ValueError("No training samples were available to calculate normalisation.")

    mean = sum_values / sample_count
    variance = np.maximum(sum_squared_values / sample_count - np.square(mean), 0.0)
    return {
        "source_split": "train",
        "sample_count": sample_count,
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    config: DatasetConfig,
    recordings: list[Recording],
    assignments: dict[str, str],
    rows: list[dict[str, object]],
    normalizer: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True)
    split_payload = {
        split_name: sorted(
            participant
            for participant, participant_split in assignments.items()
            if participant_split == split_name
        )
        for split_name in SPLIT_NAMES
    }
    summary = {
        "config": {
            "sample_rate_hz": config.sample_rate_hz,
            "vfss_frame_rate_hz": config.vfss_frame_rate_hz,
            "window_samples": config.window_samples,
            "stride_samples": config.stride_samples,
            "positive_overlap_ratio": config.positive_overlap_ratio,
            "split_seed": config.split_seed,
        },
        "recordings": len(recordings),
        "participants": len(assignments),
        "window_counts": {
            split_name: sum(row["split"] == split_name for row in rows)
            for split_name in SPLIT_NAMES
        },
        "positive_window_counts": {
            split_name: sum(
                row["split"] == split_name and row["label"] == 1
                for row in rows
            )
            for split_name in SPLIT_NAMES
        },
    }
    (output_dir / "splits.json").write_text(
        json.dumps(split_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "normalizer.json").write_text(
        json.dumps(normalizer, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "windows.csv", rows)


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    if arguments.output_dir.exists():
        if not arguments.force:
            raise FileExistsError(
                f"{arguments.output_dir} already exists. Use --force to replace generated files."
            )
        shutil.rmtree(arguments.output_dir)

    recordings = load_recordings(arguments.raw_dir, config)
    assignments = split_participants(
        {recording.participant for recording in recordings},
        config.split_ratios,
        config.split_seed,
    )
    rows = build_window_rows(recordings, assignments, config)
    normalizer = calculate_train_normalizer(arguments.raw_dir, recordings, assignments)
    write_outputs(arguments.output_dir, config, recordings, assignments, rows, normalizer)
    print(
        f"Wrote {len(rows)} windows from {len(recordings)} labeled recordings "
        f"for {len(assignments)} participants to {arguments.output_dir}."
    )


if __name__ == "__main__":
    main()
