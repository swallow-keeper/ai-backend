# Baseline model comparison

`scripts/compare_models.py` trains three window-level swallow detectors on the same Phase 2 patient-level split:

- `statistical_features`: a linear classifier over per-channel mean, standard deviation, absolute mean, maximum, and minimum.
- `one_dimensional_cnn`: the raw-signal CNN from Phase 3.
- `spectrogram_cnn`: a CNN over four-channel log-magnitude STFTs.

All candidates use the same train-only normalizer, weighted BCE loss, batch size, seed policy, and validation set.

## Selection policy

The comparison selects the candidate with the highest validation window F1. If F1 ties, it selects the lower mean batch inference time. It records validation precision, recall, F1, loss, parameter count, and inference time in `runs/comparison/comparison.json`.

This is a provisional model-selection rule, not a final performance claim. After selecting a candidate, fix its architecture and post-processing parameters using validation data, then perform event-level evaluation once on the held-out test set.

## Run

```bash
.venv/bin/python scripts/compare_models.py
```

For a bounded functional test:

```bash
.venv/bin/python scripts/compare_models.py \
  --device cpu \
  --epochs 1 \
  --max-train-samples 128 \
  --max-validation-samples 64 \
  --output-dir runs/comparison-smoke
```

Model checkpoints and result reports are stored under `runs/`, which is excluded from Git.
