import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPTS_DIRECTORY = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIRECTORY))
from event_postprocessing import EventConfig, reconstruct_events

SCRIPT_PATH = SCRIPTS_DIRECTORY / "evaluate_events.py"
SPECIFICATION = importlib.util.spec_from_file_location("evaluate_events", SCRIPT_PATH)
if SPECIFICATION is None or SPECIFICATION.loader is None:
    raise ImportError(f"Unable to load test module from {SCRIPT_PATH}")
evaluate_events = importlib.util.module_from_spec(SPECIFICATION)
sys.modules[SPECIFICATION.name] = evaluate_events
SPECIFICATION.loader.exec_module(evaluate_events)


class EventEvaluationTests(unittest.TestCase):
    def test_scores_reconstruct_a_contiguous_event(self) -> None:
        events = reconstruct_events(
            [(0, 3, 0.1), (3, 7, 0.9), (7, 10, 0.1)],
            EventConfig(0.5, 1, 0),
            stride_samples=3,
            signal_samples=10,
        )

        self.assertEqual([(event.start_sample, event.end_sample) for event in events], [(3, 7)])
        self.assertAlmostEqual(events[0].mean_score, 0.9, places=6)

    def test_matching_is_one_to_one(self) -> None:
        predictions = [
            evaluate_events.Event(100, 200),
            evaluate_events.Event(210, 300),
        ]
        targets = [
            evaluate_events.Event(110, 190),
            evaluate_events.Event(215, 295),
        ]

        matches, false_positives, false_negatives = evaluate_events.match_events(
            predictions,
            targets,
            minimum_iou=0.5,
        )

        self.assertEqual(len(matches), 2)
        self.assertEqual(false_positives, [])
        self.assertEqual(false_negatives, [])

    def test_event_metrics_counts_unmatched_predictions(self) -> None:
        metrics, _ = evaluate_events.event_metrics(
            {"file": [evaluate_events.Event(0, 10), evaluate_events.Event(20, 30)]},
            {"file": [evaluate_events.Event(0, 10)]},
            total_duration_seconds=60,
            minimum_iou=0.5,
            sample_rate_hz=10,
        )

        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 0)
