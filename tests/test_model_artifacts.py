import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch


SCRIPTS_DIRECTORY = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from model_artifacts import load_checkpoint, save_checkpoint
from model_registry import create_model


class ModelArtifactTests(unittest.TestCase):
    def test_checkpoint_round_trip_preserves_model_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory) / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "summary.json").write_text(
                json.dumps({"config": {"sample_rate_hz": 4000, "window_samples": 8000}}),
                encoding="utf-8",
            )
            (dataset_dir / "normalizer.json").write_text(
                json.dumps({"mean": [0, 0, 0, 0], "std": [1, 1, 1, 1]}),
                encoding="utf-8",
            )
            (dataset_dir / "splits.json").write_text(json.dumps({"train": ["S001"]}), encoding="utf-8")
            model = create_model("one_dimensional_cnn")
            checkpoint_path = Path(temporary_directory) / "model.pt"

            save_checkpoint(
                checkpoint_path,
                "one_dimensional_cnn",
                model,
                {"mean": [0, 0, 0, 0], "std": [1, 1, 1, 1]},
                dataset_dir,
                {"epochs": 1},
                {"f1": 0.5},
                "smoke",
            )
            loaded_model, checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))

            self.assertEqual(checkpoint["model_name"], "one_dimensional_cnn")
            self.assertEqual(checkpoint["preprocessing"]["window_samples"], 8000)
            self.assertEqual(tuple(loaded_model(torch.randn(2, 4, 8000)).shape), (2,))
