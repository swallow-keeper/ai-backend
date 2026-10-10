#!/usr/bin/env python3
# Compare window-level swallow detectors using the fixed patient-level splits.

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from train_baseline import (
    ModelConfig,
    SwallowCNN,
    SwallowWindowDataset,
    choose_device,
    class_positive_weight,
    evaluate,
    load_rows,
    set_seed,
)


MODEL_NAMES = ("statistical_features", "one_dimensional_cnn", "spectrogram_cnn")


@dataclass(frozen=True)
class ComparisonConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    seed: int
    models: tuple[str, ...]


class StatisticalFeatureClassifier(nn.Module):
    def __init__(self, input_channels: int = 4) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_channels * 5, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            (
                values.mean(dim=2),
                values.std(dim=2),
                values.abs().mean(dim=2),
                values.amax(dim=2),
                values.amin(dim=2),
            ),
            dim=1,
        )
        return self.classifier(features).squeeze(1)


class SpectrogramCNN(nn.Module):
    def __init__(self, input_channels: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.n_fft = 256
        self.hop_length = 128
        self.register_buffer("window", torch.hann_window(self.n_fft))
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, channel_count, sample_count = values.shape
        spectrogram = torch.stft(
            values.reshape(batch_size * channel_count, sample_count),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        magnitude = torch.log1p(spectrogram.abs()).reshape(
            batch_size,
            channel_count,
            spectrogram.shape[-2],
            spectrogram.shape[-1],
        )
        return self.classifier(self.features(magnitude)).squeeze(1)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare statistical, 1D CNN, and spectrogram CNN swallow detectors."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed/dataset"))
    parser.add_argument("--config", type=Path, default=Path("configs/model_comparison.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/comparison"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser.parse_args()


def load_comparison_config(path: Path) -> ComparisonConfig:
    values = json.loads(path.read_text(encoding="utf-8"))
    values["models"] = tuple(values["models"])
    config = ComparisonConfig(**values)
    unknown_models = set(config.models) - set(MODEL_NAMES)
    if unknown_models:
        raise ValueError(f"Unknown model names: {sorted(unknown_models)}")
    if not config.models:
        raise ValueError("At least one model is required.")
    return config


def create_model(model_name: str) -> nn.Module:
    if model_name == "statistical_features":
        return StatisticalFeatureClassifier()
    if model_name == "one_dimensional_cnn":
        return SwallowCNN(ModelConfig(input_channels=4, channels=(32, 64, 128), dropout=0.2))
    if model_name == "spectrogram_cnn":
        return SpectrogramCNN()
    raise ValueError(f"Unknown model name: {model_name}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def benchmark_inference(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> float:
    model.eval()
    values, _ = next(iter(loader))
    values = values.to(device)
    with torch.no_grad():
        model(values)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(3):
            model(values)
        synchronize(device)
    return (time.perf_counter() - start) * 1000 / 3


def train_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    validation_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    train_rows: list[dict[str, str]],
    config: ComparisonConfig,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]], dict[str, float], float]:
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_positive_weight(train_rows).to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None

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

        metrics = evaluate(model, validation_loader, criterion, device)
        result = {
            "epoch": epoch,
            "train_loss": total_loss / total_items,
            **{f"validation_{name}": value for name, value in metrics.items()},
        }
        history.append(result)
        print(json.dumps({"model": model_name, **result}, sort_keys=True))
        if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
            best_metrics = metrics
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    assert best_metrics is not None
    assert best_state is not None
    model.load_state_dict(best_state)
    latency_milliseconds = benchmark_inference(model, validation_loader, device)
    return model, history, best_metrics, latency_milliseconds


def main() -> None:
    arguments = parse_arguments()
    config = load_comparison_config(arguments.config)
    if arguments.epochs is not None:
        config = ComparisonConfig(**{**config.__dict__, "epochs": arguments.epochs})
    set_seed(config.seed)
    random.seed(config.seed)
    device = choose_device(arguments.device)
    summary = json.loads((arguments.dataset_dir / "summary.json").read_text(encoding="utf-8"))
    normalizer = json.loads((arguments.dataset_dir / "normalizer.json").read_text(encoding="utf-8"))
    window_samples = int(summary["config"]["window_samples"])
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
    train_dataset = SwallowWindowDataset(train_rows, arguments.raw_dir, normalizer, window_samples)
    validation_dataset = SwallowWindowDataset(
        validation_rows,
        arguments.raw_dir,
        normalizer,
        window_samples,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    model_results: list[dict[str, object]] = []

    for model_index, model_name in enumerate(config.models):
        set_seed(config.seed + model_index)
        model = create_model(model_name)
        trained_model, history, metrics, latency_milliseconds = train_model(
            model_name,
            model,
            train_loader,
            validation_loader,
            train_rows,
            config,
            device,
        )
        checkpoint_path = arguments.output_dir / f"{model_name}.pt"
        torch.save(
            {
                "model_name": model_name,
                "model_state_dict": trained_model.state_dict(),
                "normalizer": normalizer,
                "validation_metrics": metrics,
            },
            checkpoint_path,
        )
        (arguments.output_dir / f"{model_name}_history.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )
        model_results.append(
            {
                "model_name": model_name,
                "checkpoint": str(checkpoint_path),
                "parameters": parameter_count(trained_model),
                "mean_batch_inference_milliseconds": latency_milliseconds,
                "validation_metrics": metrics,
            }
        )

    selected = max(
        model_results,
        key=lambda result: (
            result["validation_metrics"]["f1"],
            -result["mean_batch_inference_milliseconds"],
        ),
    )
    output = {
        "selection_rule": "highest validation F1, then lower mean batch inference time",
        "device": str(device),
        "results": model_results,
        "selected_model": selected,
    }
    (arguments.output_dir / "comparison.json").write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected_model": selected["model_name"]}, sort_keys=True))


if __name__ == "__main__":
    main()
