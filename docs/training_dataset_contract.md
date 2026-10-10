# Swallow detector training dataset contract

`scripts/prepare_dataset.py` turns audited raw recordings into a reproducible training manifest. It does not copy or alter the raw data.

## Inputs and outputs

- Input signals: `data/raw/signals/*.npy`
- Input labels: `data/raw/labels/*.csv`
- Configuration: `configs/dataset.json`
- Generated output: `data/processed/dataset/`

The generated directory contains:

- `splits.json`: the complete patient lists for train, validation, and test.
- `windows.csv`: window boundaries, converted swallow overlap, and binary labels.
- `normalizer.json`: per-channel mean and standard deviation calculated from train signals only.
- `summary.json`: configuration and class-count summary.

The generated directory is excluded from Git. Recreate it with:

```bash
python3 scripts/prepare_dataset.py --force
```

## Split policy

Every recording belonging to one full `participant` value is assigned to exactly one split. The split is deterministic from `split_seed`, and is stratified by the `S`/`NS` filename cohort only to retain both source cohorts in each split.

`S026` and `NS026` are different participant IDs. The numeric component alone must never be used as a grouping key. Neither cohort prefix represents the binary prediction target; the target is a time window's overlap with a swallow interval.

## Label policy

- `start` and `end` are converted with `round(frame * 4000 / 60)`.
- Intervals outside the signal boundary are dropped. Their source records remain in the Phase 1 audit report.
- A window is positive when its valid swallow overlap is at least 10% of the configured 2-second window.
- Empty-label CSV files are excluded. They are not proven non-swallow recordings.
- Inference must apply the saved train-only normalizer and must not recalculate statistics from validation, test, or streaming input.
