#!/usr/bin/env python3
# Train a reproducible 1D CNN baseline for window-level swallow detection.

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int
    channels: tuple[int, ...]
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    seed: int
    model: ModelConfig


class SwallowWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        rows: list[dict[str, str]],
        raw_dir: Path,
        normalizer: dict[str, object],
        window_samples: int,
    ) -> None:
        self.rows = rows
        self.raw_dir = raw_dir
        self.mean = np.asarray(normalizer["mean"], dtype=np.float32)
        self.std = np.asarray(normalizer["std"], dtype=np.float32)
        self.window_samples = window_samples
        self._signals: dict[str, np.ndarray] = {}

        if np.any(self.std == 0):
            raise ValueError("The saved normalizer contains a zero standard deviation.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        signal = self._load_signal(row["file_name"])
        start = int(row["window_start"])
        end = int(row["window_end"])
        values = np.asarray(signal[start:end], dtype=np.float32)
        values = (values - self.mean) / self.std

        if values.shape[0] < self.window_samples:
            padding = np.zeros(
                (self.window_samples - values.shape[0], values.shape[1]),
                dtype=np.float32,
            )
            values = np.concatenate((values, padding), axis=0)
        return (
            torch.from_numpy(values.T.copy()),
            torch.tensor(float(row["label"]), dtype=torch.float32),
        )

    def _load_signal(self, file_name: str) -> np.ndarray:
        if file_name not in self._signals:
            self._signals[file_name] = np.load(
                self.raw_dir / "signals" / f"{file_name}.npy",
                mmap_mode="r",
            )
        return self._signals[file_name]


class SwallowCNN(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        input_channels = config.input_channels
        for output_channels in config.channels:
            blocks.extend(
                [
                    nn.Conv1d(input_channels, output_channels, kernel_size=7, padding=3),
                    nn.BatchNorm1d(output_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                ]
            )
            input_channels = output_channels
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(config.dropout),
            nn.Linear(input_channels, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(values)).squeeze(1)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the baseline swallow detector.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed/dataset"))
    parser.add_argument("--config", type=Path, default=Path("configs/train_baseline.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser.parse_args()


def load_training_config(path: Path) -> TrainingConfig:
    values = json.loads(path.read_text(encoding="utf-8"))
    values["model"]["channels"] = tuple(values["model"]["channels"])
    values["model"] = ModelConfig(**values["model"])
    return TrainingConfig(**values)


def choose_device(requested_device: str) -> torch.device:
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_rows(dataset_dir: Path, split: str, maximum: int | None, seed: int) -> list[dict[str, str]]:
    with (dataset_dir / "windows.csv").open(newline="", encoding="utf-8") as input_file:
        rows = [row for row in csv.DictReader(input_file) if row["split"] == split]

    if maximum is None or maximum >= len(rows):
        return rows
    generator = np.random.default_rng(seed)
    selected = generator.choice(len(rows), size=maximum, replace=False)
    return [rows[index] for index in sorted(selected)]


def binary_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    predictions = torch.sigmoid(logits) >= 0.5
    targets = targets.bool()
    true_positive = (predictions & targets).sum().item()
    false_positive = (predictions & ~targets).sum().item()
    false_negative = (~predictions & targets).sum().item()
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    logits_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    with torch.no_grad():
        for values, targets in loader:
            values = values.to(device)
            targets = targets.to(device)
            logits = model(values)
            total_loss += criterion(logits, targets).item() * values.shape[0]
            total_items += values.shape[0]
            logits_list.append(logits.cpu())
            targets_list.append(targets.cpu())

    metrics = binary_metrics(torch.cat(logits_list), torch.cat(targets_list))
    metrics["loss"] = total_loss / total_items
    return metrics


def class_positive_weight(rows: list[dict[str, str]]) -> torch.Tensor:
    positive_count = sum(row["label"] == "1" for row in rows)
    negative_count = len(rows) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Training rows must contain both positive and negative windows.")
    return torch.tensor(negative_count / positive_count, dtype=torch.float32)


def write_checkpoint(
    output_dir: Path,
    model: nn.Module,
    model_config: ModelConfig,
    normalizer: dict[str, object],
    training_config: TrainingConfig,
    validation_metrics: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_channels": model_config.input_channels,
                "channels": list(model_config.channels),
                "dropout": model_config.dropout,
            },
            "normalizer": normalizer,
            "training_config": {
                "batch_size": training_config.batch_size,
                "epochs": training_config.epochs,
                "learning_rate": training_config.learning_rate,
                "weight_decay": training_config.weight_decay,
                "seed": training_config.seed,
            },
            "validation_metrics": validation_metrics,
        },
        output_dir / "best_model.pt",
    )


def main() -> None:
    arguments = parse_arguments()
    config = load_training_config(arguments.config)
    if arguments.epochs is not None:
        config = TrainingConfig(
            **{**config.__dict__, "epochs": arguments.epochs},
        )
    set_seed(config.seed)
    device = choose_device(arguments.device)
    dataset_summary = json.loads((arguments.dataset_dir / "summary.json").read_text(encoding="utf-8"))
    normalizer = json.loads((arguments.dataset_dir / "normalizer.json").read_text(encoding="utf-8"))
    window_samples = int(dataset_summary["config"]["window_samples"])
    train_rows = load_rows(
        arguments.dataset_dir,
        "train",
        arguments.max_train_samples,
        config.seed,
    )
    validation_rows = load_rows(
        arguments.dataset_dir,
        "validation",
        arguments.max_validation_samples,
        config.seed,
    )
    train_dataset = SwallowWindowDataset(
        train_rows,
        arguments.raw_dir,
        normalizer,
        window_samples,
    )
    validation_dataset = SwallowWindowDataset(
        validation_rows,
        arguments.raw_dir,
        normalizer,
        window_samples,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size)
    model = SwallowCNN(config.model).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_positive_weight(train_rows).to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, float]] = []
    best_f1 = -1.0

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for values, targets in train_loader:
            values = values.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(values)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * values.shape[0]
            total_items += values.shape[0]

        validation_metrics = evaluate(model, validation_loader, criterion, device)
        result = {
            "epoch": epoch,
            "train_loss": total_loss / total_items,
            **{f"validation_{name}": value for name, value in validation_metrics.items()},
        }
        history.append(result)
        print(json.dumps(result, sort_keys=True))
        if validation_metrics["f1"] > best_f1:
            best_f1 = validation_metrics["f1"]
            write_checkpoint(
                arguments.output_dir,
                model,
                config.model,
                normalizer,
                config,
                validation_metrics,
            )

    (arguments.output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
