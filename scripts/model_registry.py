#!/usr/bin/env python3
# Define every supported swallow detector architecture in one registry.

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch
from torch import nn


MODEL_NAMES = ("statistical_features", "one_dimensional_cnn", "spectrogram_cnn")


class ModelSpec(TypedDict, total=False):
    input_channels: int
    channels: list[int]
    dropout: float
    n_fft: int
    hop_length: int


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int = 4
    channels: tuple[int, ...] = (32, 64, 128)
    dropout: float = 0.2


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
        self.window: torch.Tensor
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


def model_spec(model_name: str) -> ModelSpec:
    if model_name == "one_dimensional_cnn":
        return {
            "input_channels": 4,
            "channels": [32, 64, 128],
            "dropout": 0.2,
        }
    if model_name == "statistical_features":
        return {"input_channels": 4}
    if model_name == "spectrogram_cnn":
        return {"input_channels": 4, "dropout": 0.2, "n_fft": 256, "hop_length": 128}
    raise ValueError(f"Unknown model name: {model_name}")


def create_model(model_name: str, spec: ModelSpec | None = None) -> nn.Module:
    spec = spec or model_spec(model_name)
    if model_name == "one_dimensional_cnn":
        return SwallowCNN(
            ModelConfig(
                input_channels=int(spec["input_channels"]),
                channels=tuple(spec["channels"]),
                dropout=float(spec["dropout"]),
            )
        )
    if model_name == "statistical_features":
        return StatisticalFeatureClassifier(input_channels=int(spec["input_channels"]))
    if model_name == "spectrogram_cnn":
        return SpectrogramCNN(
            input_channels=int(spec["input_channels"]),
            dropout=float(spec["dropout"]),
        )
    raise ValueError(f"Unknown model name: {model_name}")
