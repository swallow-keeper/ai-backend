import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "audit_dataset.py"
SPECIFICATION = importlib.util.spec_from_file_location("audit_dataset", SCRIPT_PATH)
audit_dataset = importlib.util.module_from_spec(SPECIFICATION)
assert SPECIFICATION.loader is not None
sys.modules[SPECIFICATION.name] = audit_dataset
SPECIFICATION.loader.exec_module(audit_dataset)


class AuditDatasetTests(unittest.TestCase):
    def test_frame_to_sample_uses_vfss_frame_rate(self) -> None:
        self.assertEqual(audit_dataset.frame_to_sample(60, 4000, 60), 4000)
        self.assertEqual(audit_dataset.frame_to_sample(44, 4000, 60), 2933)

    def test_filename_pattern_preserves_full_participant_id(self) -> None:
        match = audit_dataset.FILENAME_PATTERN.fullmatch("ns026a_12")

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.group("participant"), "ns026")
        self.assertEqual(match.group("session"), "ns026a")
        self.assertEqual(match.group("run"), "12")
