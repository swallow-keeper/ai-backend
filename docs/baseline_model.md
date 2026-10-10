# Baseline swallow detector

The Phase 3 baseline is a four-channel 1D CNN that consumes one 2-second signal window and returns one swallow logit. It predicts whether the window meets the Phase 2 positive-overlap rule; it does not classify normal swallowing, aspiration, or safety.

## Training

Create the Phase 2 dataset manifest first, then train with:

```bash
.venv/bin/python scripts/train_baseline.py
```

The training command:

- uses the fixed patient-level split from `data/processed/dataset/splits.json`;
- reads the train-only channel normalizer from `normalizer.json`;
- weights positive BCE loss by the negative-to-positive train-window ratio;
- selects `runs/baseline/best_model.pt` only by validation F1;
- saves the selected model's architecture, normalizer, training settings, and validation metrics in its checkpoint.

`runs/` is excluded from Git. The held-out test split must not be used for model selection.

## CPU smoke test

Use a bounded smoke test to verify the full training path without producing a performance claim:

```bash
.venv/bin/python scripts/train_baseline.py \
  --device cpu \
  --epochs 1 \
  --max-train-samples 256 \
  --max-validation-samples 128 \
  --output-dir runs/smoke
```

The fixed decision threshold is 0.5 only for this baseline. Phase 4 will convert window scores into events and select post-processing parameters using validation data.
