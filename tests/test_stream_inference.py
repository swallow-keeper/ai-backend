import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch import nn


SCRIPTS_DIRECTORY = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIRECTORY))
SCRIPT_PATH = SCRIPTS_DIRECTORY / "stream_inference.py"
SPECIFICATION = importlib.util.spec_from_file_location("stream_inference", SCRIPT_PATH)
stream_inference = importlib.util.module_from_spec(SPECIFICATION)
assert SPECIFICATION.loader is not None
sys.modules[SPECIFICATION.name] = stream_inference
SPECIFICATION.loader.exec_module(stream_inference)


class MeanLogitModel(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values.mean(dim=(1, 2))


class StreamInferenceTests(unittest.TestCase):
    def test_assembler_merges_positive_windows_and_flushes(self) -> None:
        assembler = stream_inference.OnlineEventAssembler(score_threshold=0.5, merge_gap_samples=100)

        self.assertEqual(assembler.add_score(0, 8_000, 0.8, 8_000), [])
        self.assertEqual(assembler.add_score(1_000, 9_000, 0.9, 9_000), [])
        events = assembler.flush(10_000)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].start_sample, 0)
        self.assertEqual(events[0].end_sample, 9_000)
        self.assertEqual(events[0].emitted_at_sample, 10_000)
        self.assertAlmostEqual(events[0].mean_score, 0.85)

    def test_detector_processes_overlapping_windows(self) -> None:
        config = stream_inference.StreamingConfig(
            sample_rate_hz=4,
            window_seconds=1.0,
            stride_seconds=0.5,
            score_threshold=0.5,
            merge_gap_seconds=0.0,
            chunk_samples=2,
        )
        detector = stream_inference.StreamingSwallowDetector(
            MeanLogitModel(),
            np.zeros(4),
            np.ones(4),
            config,
            torch.device("cpu"),
        )

        detector.ingest(np.full((2, 4), 10.0, dtype=np.float32))
        detector.ingest(np.full((2, 4), 10.0, dtype=np.float32))
        detector.ingest(np.zeros((2, 4), dtype=np.float32))
        events = detector.flush()

        self.assertEqual(detector.window_count, 2)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].start_sample, 0)
        self.assertEqual(events[0].end_sample, 6)
