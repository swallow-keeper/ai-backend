import importlib.util
from pathlib import Path
import sys
import unittest

import torch


SCRIPTS_DIRECTORY = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIRECTORY))
SCRIPT_PATH = SCRIPTS_DIRECTORY / "train_baseline.py"
SPECIFICATION = importlib.util.spec_from_file_location("train_baseline", SCRIPT_PATH)
train_baseline = importlib.util.module_from_spec(SPECIFICATION)
assert SPECIFICATION.loader is not None
sys.modules[SPECIFICATION.name] = train_baseline
SPECIFICATION.loader.exec_module(train_baseline)


class TrainBaselineTests(unittest.TestCase):
    def test_model_outputs_one_logit_per_window(self) -> None:
        config = train_baseline.ModelConfig(
            input_channels=4,
            channels=(8, 16),
            dropout=0.2,
        )
        model = train_baseline.SwallowCNN(config)

        logits = model(torch.randn(3, 4, 8_000))

        self.assertEqual(tuple(logits.shape), (3,))

    def test_binary_metrics_for_perfect_predictions(self) -> None:
        metrics = train_baseline.binary_metrics(
            torch.tensor([10.0, -10.0, 5.0, -5.0]),
            torch.tensor([1.0, 0.0, 1.0, 0.0]),
        )

        self.assertEqual(metrics, {"precision": 1.0, "recall": 1.0, "f1": 1.0})
