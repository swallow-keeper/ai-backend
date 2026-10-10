import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPTS_DIRECTORY = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIRECTORY))
SCRIPT_PATH = SCRIPTS_DIRECTORY / "prepare_dataset.py"
SPECIFICATION = importlib.util.spec_from_file_location("prepare_dataset", SCRIPT_PATH)
prepare_dataset = importlib.util.module_from_spec(SPECIFICATION)
assert SPECIFICATION.loader is not None
sys.modules[SPECIFICATION.name] = prepare_dataset
SPECIFICATION.loader.exec_module(prepare_dataset)


class PrepareDatasetTests(unittest.TestCase):
    def test_participants_are_assigned_to_one_split(self) -> None:
        assignments = prepare_dataset.split_participants(
            {"S001", "S002", "S003", "S004", "S005", "S006", "S007", "NS001", "NS002", "NS003", "NS004", "NS005", "NS006", "NS007"},
            {"train": 0.7, "validation": 0.15, "test": 0.15},
            seed=1,
        )

        self.assertEqual(set(assignments), {"S001", "S002", "S003", "S004", "S005", "S006", "S007", "NS001", "NS002", "NS003", "NS004", "NS005", "NS006", "NS007"})
        self.assertEqual(set(assignments.values()), {"train", "validation", "test"})

    def test_final_window_reaches_signal_end(self) -> None:
        self.assertEqual(
            prepare_dataset.window_starts(10_500, window_samples=8_000, stride_samples=1_000),
            [0, 1_000, 2_000, 2_500],
        )

    def test_overlapping_interval_marks_positive_window(self) -> None:
        recording = prepare_dataset.Recording(
            file_name="s001a_1",
            participant="S001",
            signal_samples=8_000,
            channels=4,
            intervals=((700, 1_700),),
        )
        config = prepare_dataset.DatasetConfig(
            sample_rate_hz=4_000,
            vfss_frame_rate_hz=60,
            window_seconds=2.0,
            stride_seconds=2.0,
            positive_overlap_ratio=0.1,
            split_seed=1,
            split_ratios={"train": 0.7, "validation": 0.15, "test": 0.15},
        )

        row = prepare_dataset.build_window_rows([recording], {"S001": "train"}, config)[0]

        self.assertEqual(row["positive_samples"], 1_000)
        self.assertEqual(row["label"], 1)
