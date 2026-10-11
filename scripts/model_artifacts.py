#!/usr/bin/env python3
# Save and load portable model artifacts with reproducibility metadata.

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypedDict, cast

import torch
from torch import nn

from model_registry import create_model, model_spec


CHECKPOINT_SCHEMA_VERSION = 1


class PreprocessingMetadata(TypedDict):
    input_layout: str
    padding_policy: str
    sample_rate_hz: int
    window_samples: int


class DatasetSummaryConfig(TypedDict):
    sample_rate_hz: int
    window_samples: int


class DatasetSummary(TypedDict):
    config: DatasetSummaryConfig


class DatasetMetadata(TypedDict):
    dataset_dir: str
    summary: DatasetSummary
    summary_sha256: str
    normalizer_sha256: str
    splits_sha256: str


class Checkpoint(TypedDict):
    schema_version: int
    model_name: str
    model_spec: dict[str, object]
    model_state_dict: dict[str, torch.Tensor]
    normalizer: dict[str, object]
    preprocessing: PreprocessingMetadata
    dataset: DatasetMetadata
    training_config: dict[str, object]
    validation_metrics: dict[str, float]
    run_kind: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_metadata(dataset_dir: Path) -> DatasetMetadata:
    summary_path = dataset_dir / "summary.json"
    normalizer_path = dataset_dir / "normalizer.json"
    splits_path = dataset_dir / "splits.json"
    summary = cast(DatasetSummary, json.loads(summary_path.read_text(encoding="utf-8")))
    return {
        "dataset_dir": str(dataset_dir),
        "summary": summary,
        "summary_sha256": file_sha256(summary_path),
        "normalizer_sha256": file_sha256(normalizer_path),
        "splits_sha256": file_sha256(splits_path),
    }


def save_checkpoint(
    path: Path,
    model_name: str,
    model: nn.Module,
    normalizer: dict[str, object],
    dataset_dir: Path,
    training_config: dict[str, object],
    validation_metrics: dict[str, float],
    run_kind: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = dataset_metadata(dataset_dir)
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_name": model_name,
            "model_spec": model_spec(model_name),
            "model_state_dict": model.state_dict(),
            "normalizer": normalizer,
            "preprocessing": {
                "input_layout": "channels_first",
                "padding_policy": "zero_after_train_normalization",
                "sample_rate_hz": dataset["summary"]["config"]["sample_rate_hz"],
                "window_samples": dataset["summary"]["config"]["window_samples"],
            },
            "dataset": dataset,
            "training_config": training_config,
            "validation_metrics": validation_metrics,
            "run_kind": run_kind,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, Checkpoint]:
    checkpoint = cast(Checkpoint, torch.load(path, map_location="cpu", weights_only=False))
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported legacy checkpoint. Train a new metadata-complete checkpoint.")
    model = create_model(checkpoint["model_name"], checkpoint["model_spec"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint
