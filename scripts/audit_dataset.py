#!/usr/bin/env python3
# Audit the raw swallowing dataset without changing its source files.

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_SAMPLE_RATE_HZ = 4000
DEFAULT_VFSS_FRAME_RATE_HZ = 60
FILENAME_PATTERN = re.compile(r"^(?P<session>(?P<participant>(?:ns|s)\d{3})[a-z])_(?P<run>\d+)$")
EXPECTED_LABEL_COLUMNS = (
    "participant",
    "file_num",
    "swallow_num",
    "start",
    "end",
    "duration",
    "fnames",
)


@dataclass(frozen=True)
class InvalidInterval:
    file_name: str
    swallow_num: str
    start_frame: float
    end_frame: float
    start_sample: int
    end_sample: int
    signal_samples: int
    reason: str


@dataclass
class FileAudit:
    file_name: str
    participant: str
    session: str
    run: int
    signal_samples: int
    channels: int
    dtype: str
    label_rows: int
    valid_events: int
    invalid_events: int
    empty_labels: bool


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit raw swallowing signals and VFSS labels.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/audit"))
    parser.add_argument("--report", type=Path, default=Path("docs/dataset_audit.md"))
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("data/processed/audit/plots"),
        help="Directory for representative label-alignment plots.",
    )
    parser.add_argument("--sample-rate-hz", type=int, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument("--vfss-frame-rate-hz", type=int, default=DEFAULT_VFSS_FRAME_RATE_HZ)
    return parser.parse_args()


def frame_to_sample(frame: float, sample_rate_hz: int, frame_rate_hz: int) -> int:
    return round(frame * sample_rate_hz / frame_rate_hz)


def read_label_rows(label_path: Path) -> list[dict[str, str]]:
    with label_path.open(newline="", encoding="utf-8") as label_file:
        reader = csv.DictReader(label_file)
        if reader.fieldnames != list(EXPECTED_LABEL_COLUMNS):
            raise ValueError(
                f"{label_path}: expected columns {EXPECTED_LABEL_COLUMNS}, found {reader.fieldnames}"
            )
        return list(reader)


def calculate_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("utf-8"))
        digest.update(str(path.stat().st_mtime_ns).encode("utf-8"))
    return digest.hexdigest()


def audit_dataset(
    raw_dir: Path,
    sample_rate_hz: int,
    frame_rate_hz: int,
) -> tuple[list[FileAudit], list[InvalidInterval], dict[str, object]]:
    signals_dir = raw_dir / "signals"
    labels_dir = raw_dir / "labels"
    signal_paths = sorted(signals_dir.glob("*.npy"))
    label_paths = sorted(labels_dir.glob("*.csv"))

    if not signal_paths or not label_paths:
        raise FileNotFoundError("Expected non-empty signals/*.npy and labels/*.csv directories.")

    signal_stems = {path.stem for path in signal_paths}
    label_stems = {path.stem for path in label_paths}
    if signal_stems != label_stems:
        missing_labels = sorted(signal_stems - label_stems)
        missing_signals = sorted(label_stems - signal_stems)
        raise ValueError(
            f"Signal/label stems differ. Missing labels: {missing_labels[:5]}; "
            f"missing signals: {missing_signals[:5]}"
        )

    files: list[FileAudit] = []
    invalid_intervals: list[InvalidInterval] = []
    participant_event_counts: Counter[str] = Counter()
    participant_file_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    shape_counts: Counter[int] = Counter()

    for signal_path in signal_paths:
        match = FILENAME_PATTERN.fullmatch(signal_path.stem)
        if match is None:
            raise ValueError(f"Unexpected filename format: {signal_path.name}")

        signal = np.load(signal_path, mmap_mode="r")
        if signal.ndim != 2:
            raise ValueError(f"{signal_path}: expected a 2D signal array, found {signal.shape}")

        label_rows = read_label_rows(labels_dir / f"{signal_path.stem}.csv")
        participant = match.group("participant").upper()
        session = match.group("session").lower()
        run = int(match.group("run"))
        valid_events = 0
        invalid_events = 0

        for row in label_rows:
            if row["participant"].upper() != participant:
                raise ValueError(
                    f"{signal_path.stem}: filename participant {participant} "
                    f"does not match label participant {row['participant']}"
                )

            start_frame = float(row["start"])
            end_frame = float(row["end"])
            start_sample = frame_to_sample(start_frame, sample_rate_hz, frame_rate_hz)
            end_sample = frame_to_sample(end_frame, sample_rate_hz, frame_rate_hz)
            problems: list[str] = []
            if end_frame <= start_frame:
                problems.append("non_positive_frame_duration")
            if start_sample < 0:
                problems.append("negative_start_sample")
            if end_sample > signal.shape[0]:
                problems.append("end_sample_out_of_bounds")

            if problems:
                invalid_events += 1
                invalid_intervals.append(
                    InvalidInterval(
                        file_name=signal_path.stem,
                        swallow_num=row["swallow_num"],
                        start_frame=start_frame,
                        end_frame=end_frame,
                        start_sample=start_sample,
                        end_sample=end_sample,
                        signal_samples=signal.shape[0],
                        reason=",".join(problems),
                    )
                )
            else:
                valid_events += 1

        files.append(
            FileAudit(
                file_name=signal_path.stem,
                participant=participant,
                session=session,
                run=run,
                signal_samples=signal.shape[0],
                channels=signal.shape[1],
                dtype=str(signal.dtype),
                label_rows=len(label_rows),
                valid_events=valid_events,
                invalid_events=invalid_events,
                empty_labels=not label_rows,
            )
        )
        participant_event_counts[participant] += valid_events
        participant_file_counts[participant] += 1
        dtype_counts[str(signal.dtype)] += 1
        shape_counts[signal.shape[1]] += 1

    summary: dict[str, object] = {
        "raw_data_signature": calculate_sha256([*signal_paths, *label_paths]),
        "paired_files": len(files),
        "participants": len(participant_file_counts),
        "events_total": sum(file.label_rows for file in files),
        "events_valid": sum(file.valid_events for file in files),
        "events_invalid": len(invalid_intervals),
        "empty_label_files": sum(file.empty_labels for file in files),
        "sample_rate_hz": sample_rate_hz,
        "vfss_frame_rate_hz": frame_rate_hz,
        "sample_conversion": "round(frame * sample_rate_hz / vfss_frame_rate_hz)",
        "signal_dtype_counts": dict(dtype_counts),
        "channel_count_distribution": dict(shape_counts),
        "participant_file_counts": dict(sorted(participant_file_counts.items())),
        "participant_event_counts": dict(sorted(participant_event_counts.items())),
    }
    return files, invalid_intervals, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    report_path: Path,
    summary: dict[str, object],
    files: list[FileAudit],
    invalid_intervals: list[InvalidInterval],
) -> None:
    lengths = np.array([file.signal_samples for file in files])
    event_counts = np.array([file.valid_events for file in files])
    lines = [
        "# Raw swallowing dataset audit",
        "",
        "This report is generated by `scripts/audit_dataset.py`; do not edit it by hand.",
        "",
        "## Dataset identity",
        "",
        f"- Raw-data signature: `{summary['raw_data_signature']}`",
        f"- Paired signal/label files: {summary['paired_files']}",
        f"- Participants: {summary['participants']}",
        f"- Signal arrays: `{summary['signal_dtype_counts']}`, "
        f"channel counts `{summary['channel_count_distribution']}`",
        "",
        "## Label time conversion",
        "",
        f"- Signal sampling rate: {summary['sample_rate_hz']} Hz",
        f"- VFSS frame rate: {summary['vfss_frame_rate_hz']} Hz",
        f"- Conversion: `{summary['sample_conversion']}`",
        "- `start` and `end` are VFSS-frame positions, not signal row indices.",
        "",
        "## Audit results",
        "",
        f"- Label rows: {summary['events_total']}",
        f"- Valid events after conversion: {summary['events_valid']}",
        f"- Invalid events: {summary['events_invalid']}",
        f"- Empty label CSV files: {summary['empty_label_files']}",
        f"- Signal duration range: {lengths.min() / summary['sample_rate_hz']:.2f}–"
        f"{lengths.max() / summary['sample_rate_hz']:.2f} seconds",
        f"- Median signal duration: {np.median(lengths) / summary['sample_rate_hz']:.2f} seconds",
        f"- Median valid events per clip: {np.median(event_counts):.0f}",
        "",
        "## Training safeguards",
        "",
        "- Split by the complete `participant` value, never by individual file or numeric ID alone.",
        "- Treat `S026` and `NS026` as distinct IDs; both prefixes contain swallow annotations.",
        "- Exclude empty-label files from negative training until their annotation status is verified.",
        "- Exclude invalid intervals by default and retain `invalid_intervals.csv` for traceability.",
        "",
        "## Invalid intervals",
        "",
    ]
    if invalid_intervals:
        lines.extend(
            [
                "| File | Swallow | Start frame | End frame | Start sample | End sample | Signal samples | Reason |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for item in invalid_intervals:
            lines.append(
                f"| {item.file_name} | {item.swallow_num} | {item.start_frame:g} | "
                f"{item.end_frame:g} | {item.start_sample} | {item.end_sample} | "
                f"{item.signal_samples} | {item.reason} |"
            )
    else:
        lines.append("No invalid intervals were found.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_signal_with_intervals(
    signal_path: Path,
    label_path: Path,
    output_path: Path,
    sample_rate_hz: int,
    frame_rate_hz: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    signal = np.load(signal_path, mmap_mode="r")
    label_rows = read_label_rows(label_path)
    time_seconds = np.arange(signal.shape[0]) / sample_rate_hz
    figure, axes = plt.subplots(signal.shape[1], 1, figsize=(14, 8), sharex=True)
    channel_names = ("microphone", "accelerometer_ap", "accelerometer_si", "accelerometer_ml")

    for channel_index, axis in enumerate(axes):
        axis.plot(time_seconds, signal[:, channel_index], linewidth=0.5)
        axis.set_ylabel(channel_names[channel_index])
        for row in label_rows:
            start = frame_to_sample(float(row["start"]), sample_rate_hz, frame_rate_hz)
            end = frame_to_sample(float(row["end"]), sample_rate_hz, frame_rate_hz)
            axis.axvspan(
                max(start, 0) / sample_rate_hz,
                min(end, signal.shape[0]) / sample_rate_hz,
                alpha=0.2,
                color="tab:green",
            )

    axes[-1].set_xlabel("Time (seconds)")
    figure.suptitle(f"{signal_path.stem}: green spans are converted swallow labels")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def generate_representative_plots(
    raw_dir: Path,
    plot_dir: Path,
    files: list[FileAudit],
    sample_rate_hz: int,
    frame_rate_hz: int,
) -> list[Path]:
    positive_file = next(file for file in files if file.valid_events > 0)
    empty_file = next(file for file in files if file.empty_labels)
    selected_files = (positive_file, empty_file)
    paths: list[Path] = []

    for file in selected_files:
        output_path = plot_dir / f"{file.file_name}.png"
        plot_signal_with_intervals(
            raw_dir / "signals" / f"{file.file_name}.npy",
            raw_dir / "labels" / f"{file.file_name}.csv",
            output_path,
            sample_rate_hz,
            frame_rate_hz,
        )
        paths.append(output_path)
    return paths


def main() -> None:
    arguments = parse_arguments()
    files, invalid_intervals, summary = audit_dataset(
        raw_dir=arguments.raw_dir,
        sample_rate_hz=arguments.sample_rate_hz,
        frame_rate_hz=arguments.vfss_frame_rate_hz,
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(arguments.output_dir / "files.csv", [asdict(file) for file in files])
    write_csv(
        arguments.output_dir / "invalid_intervals.csv",
        [asdict(interval) for interval in invalid_intervals],
    )
    (arguments.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(arguments.report, summary, files, invalid_intervals)
    plot_paths = generate_representative_plots(
        arguments.raw_dir,
        arguments.plot_dir,
        files,
        arguments.sample_rate_hz,
        arguments.vfss_frame_rate_hz,
    )
    print(f"Audited {summary['paired_files']} paired files and wrote {arguments.report}.")
    print(f"Representative plots: {', '.join(str(path) for path in plot_paths)}")


if __name__ == "__main__":
    main()
