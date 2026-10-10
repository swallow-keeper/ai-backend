#!/usr/bin/env python3
# Replay signal chunks through a baseline model and emit online swallow events.

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train_baseline import ModelConfig, SwallowCNN, choose_device


@dataclass(frozen=True)
class StreamingConfig:
    sample_rate_hz: int
    window_seconds: float
    stride_seconds: float
    score_threshold: float
    merge_gap_seconds: float
    chunk_samples: int

    @property
    def window_samples(self) -> int:
        return round(self.window_seconds * self.sample_rate_hz)

    @property
    def stride_samples(self) -> int:
        return round(self.stride_seconds * self.sample_rate_hz)

    @property
    def merge_gap_samples(self) -> int:
        return round(self.merge_gap_seconds * self.sample_rate_hz)


@dataclass(frozen=True)
class StreamEvent:
    start_sample: int
    end_sample: int
    emitted_at_sample: int
    mean_score: float


class OnlineEventAssembler:
    def __init__(self, score_threshold: float, merge_gap_samples: int) -> None:
        self.score_threshold = score_threshold
        self.merge_gap_samples = merge_gap_samples
        self._active_start: int | None = None
        self._active_end: int | None = None
        self._score_sum = 0.0
        self._score_count = 0

    def add_score(
        self,
        start_sample: int,
        end_sample: int,
        score: float,
        emitted_at_sample: int,
    ) -> list[StreamEvent]:
        emitted: list[StreamEvent] = []
        if score >= self.score_threshold:
            if self._active_start is None:
                self._active_start = start_sample
            self._active_end = max(self._active_end or end_sample, end_sample)
            self._score_sum += score
            self._score_count += 1
            return emitted

        if self._active_end is not None and start_sample - self._active_end > self.merge_gap_samples:
            emitted.append(self._close_event(emitted_at_sample))
        return emitted

    def flush(self, emitted_at_sample: int) -> list[StreamEvent]:
        if self._active_end is None:
            return []
        return [self._close_event(emitted_at_sample)]

    def _close_event(self, emitted_at_sample: int) -> StreamEvent:
        assert self._active_start is not None
        assert self._active_end is not None
        event = StreamEvent(
            start_sample=self._active_start,
            end_sample=self._active_end,
            emitted_at_sample=emitted_at_sample,
            mean_score=self._score_sum / self._score_count,
        )
        self._active_start = None
        self._active_end = None
        self._score_sum = 0.0
        self._score_count = 0
        return event


class StreamingSwallowDetector:
    def __init__(
        self,
        model: nn.Module,
        mean: np.ndarray,
        std: np.ndarray,
        config: StreamingConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device).eval()
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.config = config
        self.device = device
        self.assembler = OnlineEventAssembler(config.score_threshold, config.merge_gap_samples)
        self.buffer = np.empty((0, len(mean)), dtype=np.float32)
        self.buffer_start = 0
        self.received_samples = 0
        self.next_window_start = 0
        self.window_count = 0

        if np.any(self.std == 0):
            raise ValueError("The saved normalizer contains a zero standard deviation.")

    def ingest(self, values: np.ndarray) -> list[StreamEvent]:
        if values.ndim != 2 or values.shape[1] != len(self.mean):
            raise ValueError(f"Expected chunk shape (samples, {len(self.mean)}), found {values.shape}.")
        self.buffer = np.concatenate((self.buffer, values.astype(np.float32)), axis=0)
        self.received_samples += values.shape[0]
        emitted: list[StreamEvent] = []

        while self.next_window_start + self.config.window_samples <= self.received_samples:
            start_index = self.next_window_start - self.buffer_start
            end_index = start_index + self.config.window_samples
            window = self.buffer[start_index:end_index]
            score = self._score_window(window)
            emitted.extend(
                self.assembler.add_score(
                    self.next_window_start,
                    self.next_window_start + self.config.window_samples,
                    score,
                    self.received_samples,
                )
            )
            self.next_window_start += self.config.stride_samples
            self.window_count += 1

        self._discard_consumed_prefix()
        return emitted

    def flush(self) -> list[StreamEvent]:
        return self.assembler.flush(self.received_samples)

    def _score_window(self, window: np.ndarray) -> float:
        normalized = (window - self.mean) / self.std
        tensor = torch.from_numpy(normalized.T.copy()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return float(torch.sigmoid(self.model(tensor)).item())

    def _discard_consumed_prefix(self) -> None:
        discard_samples = self.next_window_start - self.buffer_start
        if discard_samples <= 0:
            return
        self.buffer = self.buffer[discard_samples:]
        self.buffer_start = self.next_window_start


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a signal as streaming input to the baseline detector.")
    parser.add_argument("--signal", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/baseline/best_model.pt"))
    parser.add_argument("--config", type=Path, default=Path("configs/streaming_inference.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/streaming"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser.parse_args()


def load_config(path: Path) -> StreamingConfig:
    config = StreamingConfig(**json.loads(path.read_text(encoding="utf-8")))
    if config.window_samples <= 0 or config.stride_samples <= 0 or config.chunk_samples <= 0:
        raise ValueError("Window, stride, and chunk sizes must be positive.")
    if not 0 <= config.score_threshold <= 1:
        raise ValueError("score_threshold must be in [0, 1].")
    return config


def load_baseline_model(checkpoint_path: Path, device: torch.device) -> tuple[SwallowCNN, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_config" not in checkpoint:
        raise ValueError("Checkpoint is not a Phase 3 baseline checkpoint.")
    model_config = ModelConfig(
        input_channels=checkpoint["model_config"]["input_channels"],
        channels=tuple(checkpoint["model_config"]["channels"]),
        dropout=checkpoint["model_config"]["dropout"],
    )
    model = SwallowCNN(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device), checkpoint["normalizer"]


def event_payload(event: StreamEvent, sample_rate_hz: int) -> dict[str, float | int]:
    return {
        **asdict(event),
        "start_seconds": event.start_sample / sample_rate_hz,
        "end_seconds": event.end_sample / sample_rate_hz,
        "emitted_at_seconds": event.emitted_at_sample / sample_rate_hz,
        "algorithmic_latency_seconds": (event.emitted_at_sample - event.end_sample) / sample_rate_hz,
    }


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    device = choose_device(arguments.device)
    model, normalizer = load_baseline_model(arguments.checkpoint, device)
    signal = np.load(arguments.signal, mmap_mode="r")
    detector = StreamingSwallowDetector(
        model,
        np.asarray(normalizer["mean"]),
        np.asarray(normalizer["std"]),
        config,
        device,
    )
    events: list[StreamEvent] = []
    inference_start = time.perf_counter()
    for start in range(0, signal.shape[0], config.chunk_samples):
        events.extend(detector.ingest(signal[start : start + config.chunk_samples]))
    events.extend(detector.flush())
    elapsed_seconds = time.perf_counter() - inference_start
    output = {
        "signal": str(arguments.signal),
        "device": str(device),
        "signal_samples": int(signal.shape[0]),
        "chunk_samples": config.chunk_samples,
        "window_count": detector.window_count,
        "wall_inference_seconds": elapsed_seconds,
        "wall_seconds_per_window": elapsed_seconds / detector.window_count if detector.window_count else 0.0,
        "events": [event_payload(event, config.sample_rate_hz) for event in events],
    }
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.output_dir / "replay_report.json").write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: output[key] for key in ("window_count", "wall_inference_seconds", "events")}))


if __name__ == "__main__":
    main()
