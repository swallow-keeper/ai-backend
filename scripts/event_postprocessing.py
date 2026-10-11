#!/usr/bin/env python3
# Share causal score aggregation and event reconstruction across offline and streaming paths.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Event:
    start_sample: int
    end_sample: int
    emitted_at_sample: int = 0
    mean_score: float = 0.0


@dataclass(frozen=True)
class EventConfig:
    score_threshold: float
    minimum_event_samples: int
    merge_gap_samples: int


class CausalEventPostprocessor:
    def __init__(self, config: EventConfig, stride_samples: int) -> None:
        self.config = config
        self.stride_samples = stride_samples
        self._buffer_start = 0
        self._score_sums = np.empty(0, dtype=np.float32)
        self._score_counts = np.empty(0, dtype=np.uint16)
        self._event_start: int | None = None
        self._event_end: int | None = None
        self._event_score_sum = 0.0
        self._event_score_count = 0

    def add_window_score(
        self,
        start_sample: int,
        end_sample: int,
        score: float,
        available_at_sample: int,
    ) -> list[Event]:
        self._extend_to(end_sample)
        start_index = start_sample - self._buffer_start
        end_index = end_sample - self._buffer_start
        self._score_sums[start_index:end_index] += score
        self._score_counts[start_index:end_index] += 1
        return self._finalize_until(
            min(start_sample + self.stride_samples, end_sample),
            available_at_sample,
        )

    def flush(self, available_at_sample: int) -> list[Event]:
        emitted = self._finalize_until(self._buffer_start + len(self._score_sums), available_at_sample)
        if self._event_end is not None:
            emitted.append(self._close_event(available_at_sample))
        return emitted

    def _extend_to(self, end_sample: int) -> None:
        required_size = end_sample - self._buffer_start
        if required_size > len(self._score_sums):
            extension = required_size - len(self._score_sums)
            self._score_sums = np.pad(self._score_sums, (0, extension))
            self._score_counts = np.pad(self._score_counts, (0, extension))

    def _finalize_until(self, end_sample: int, available_at_sample: int) -> list[Event]:
        count = end_sample - self._buffer_start
        if count <= 0:
            return []
        scores = np.divide(
            self._score_sums[:count],
            self._score_counts[:count],
            out=np.zeros(count, dtype=np.float32),
            where=self._score_counts[:count] > 0,
        )
        active = scores >= self.config.score_threshold
        emitted = self._consume_active_spans(scores, active, self._buffer_start, available_at_sample)
        self._score_sums = self._score_sums[count:]
        self._score_counts = self._score_counts[count:]
        self._buffer_start = end_sample
        return emitted

    def _consume_active_spans(
        self,
        scores: np.ndarray,
        active: np.ndarray,
        start_sample: int,
        available_at_sample: int,
    ) -> list[Event]:
        emitted: list[Event] = []
        changes = np.flatnonzero(np.diff(active.astype(np.int8))) + 1
        boundaries = np.concatenate(([0], changes, [len(active)]))
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            if not active[start]:
                continue
            span_start = start_sample + int(start)
            span_end = start_sample + int(end)
            if self._event_end is None:
                self._start_event(span_start, span_end, scores[start:end])
            elif span_start - self._event_end <= self.config.merge_gap_samples:
                self._extend_event(span_end, scores[start:end])
            else:
                emitted.append(self._close_event(available_at_sample))
                self._start_event(span_start, span_end, scores[start:end])

        if self._event_end is not None and start_sample + len(active) > self._event_end + self.config.merge_gap_samples:
            emitted.append(self._close_event(available_at_sample))
        return [event for event in emitted if event.end_sample - event.start_sample >= self.config.minimum_event_samples]

    def _start_event(self, start_sample: int, end_sample: int, scores: np.ndarray) -> None:
        self._event_start = start_sample
        self._event_end = end_sample
        self._event_score_sum = float(scores.sum())
        self._event_score_count = len(scores)

    def _extend_event(self, end_sample: int, scores: np.ndarray) -> None:
        self._event_end = end_sample
        self._event_score_sum += float(scores.sum())
        self._event_score_count += len(scores)

    def _close_event(self, emitted_at_sample: int) -> Event:
        assert self._event_start is not None
        assert self._event_end is not None
        event = Event(
            self._event_start,
            self._event_end,
            emitted_at_sample,
            self._event_score_sum / self._event_score_count,
        )
        self._event_start = None
        self._event_end = None
        self._event_score_sum = 0.0
        self._event_score_count = 0
        return event


def reconstruct_events(
    window_scores: list[tuple[int, int, float]],
    config: EventConfig,
    stride_samples: int,
    signal_samples: int,
) -> list[Event]:
    processor = CausalEventPostprocessor(config, stride_samples)
    events: list[Event] = []
    for start, end, score in window_scores:
        events.extend(processor.add_window_score(start, end, score, end))
    events.extend(processor.flush(signal_samples))
    return events
