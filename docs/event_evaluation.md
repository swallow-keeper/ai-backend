# Event-level evaluation

Window classification metrics do not measure whether a system detects individual swallows. `scripts/evaluate_events.py` converts held-out test-window scores into events, then compares the reconstructed events with the original converted VFSS intervals.

## Reconstruction rule

1. Each window probability is written over its original signal interval.
2. Scores from overlapping windows are averaged per signal sample.
3. Samples at or above the configured threshold become candidate event spans.
4. Candidate spans separated by at most `merge_gap_seconds` are merged.
5. Spans shorter than `minimum_event_seconds` are removed.

The default post-processing configuration is in `configs/event_evaluation.json`. Tune it only using validation data; preserve the chosen configuration before running test evaluation.

## Event matching

Predicted and reference events are matched one-to-one by descending intersection-over-union (IoU). A pair is a true positive only if its IoU is at least `minimum_iou`. Unmatched predictions are false positives; unmatched references are false negatives.

The evaluator reports event precision, recall, F1, false positives per minute, matched-event mean IoU, and mean onset/offset errors. These metrics quantify swallow-event timing only. They do not assess aspiration, safety, or any clinical outcome.

## Run

```bash
.venv/bin/python scripts/evaluate_events.py \
  --checkpoint runs/baseline/best_model.pt \
  --device cpu
```

Reports are saved under `runs/evaluation/` and excluded from Git.
