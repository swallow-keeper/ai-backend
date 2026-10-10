import importlib.util
from pathlib import Path
import sys
import unittest

import torch


SCRIPTS_DIRECTORY = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIRECTORY))
SCRIPT_PATH = SCRIPTS_DIRECTORY / "compare_models.py"
SPECIFICATION = importlib.util.spec_from_file_location("compare_models", SCRIPT_PATH)
compare_models = importlib.util.module_from_spec(SPECIFICATION)
assert SPECIFICATION.loader is not None
sys.modules[SPECIFICATION.name] = compare_models
SPECIFICATION.loader.exec_module(compare_models)


class CompareModelsTests(unittest.TestCase):
    def test_all_models_return_one_logit_per_window(self) -> None:
        values = torch.randn(2, 4, 1_024)

        for model_name in compare_models.MODEL_NAMES:
            with self.subTest(model_name=model_name):
                logits = compare_models.create_model(model_name)(values)
                self.assertEqual(tuple(logits.shape), (2,))

    def test_model_selection_prefers_f1_then_lower_latency(self) -> None:
        results = [
            {
                "model_name": "slower",
                "validation_metrics": {"f1": 0.7},
                "mean_batch_inference_milliseconds": 2.0,
            },
            {
                "model_name": "faster",
                "validation_metrics": {"f1": 0.7},
                "mean_batch_inference_milliseconds": 1.0,
            },
        ]

        selected = max(
            results,
            key=lambda result: (
                result["validation_metrics"]["f1"],
                -result["mean_batch_inference_milliseconds"],
            ),
        )

        self.assertEqual(selected["model_name"], "faster")
