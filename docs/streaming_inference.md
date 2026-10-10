# Streaming swallow inference

`scripts/stream_inference.py` replays a stored `(samples, 4)` signal as fixed-size chunks. It is the integration test path before replacing the file reader with ESP32-S3 transport.

## Streaming contract

The upstream collector must provide:

- ordered `float` samples with one timestamp per packet or an unambiguous sample rate;
- four channels in the model's recorded order;
- a configured 4 kHz rate until a new model is trained for another rate;
- monotonic sample offsets, including an explicit marker for packet loss or discontinuity.

The detector receives chunks, creates overlapping 2-second windows every 0.25 seconds, normalizes with the train-only checkpoint values, and produces a score. Consecutive score-positive windows are merged into an event. An active event is emitted after enough negative window time or at stream end.

The event output contains start/end sample positions, emission position, score, and algorithmic latency. It only means that the model detected a swallow-like signal pattern; it is not an aspiration, safety, or clinical judgment.

## Replay

```bash
.venv/bin/python scripts/stream_inference.py \
  --signal data/raw/signals/s048a_1.npy \
  --checkpoint runs/baseline/best_model.pt \
  --device cpu
```

`runs/streaming/replay_report.json` records window count, wall inference time, and emitted events. Replay latency measures local model execution only. Network, ADC, and sensor buffering latency must be measured after actual hardware transport is connected.

The first integration supports Phase 3 baseline checkpoints. If Phase 5 selects another architecture, add its checkpoint loader before deployment; do not silently run a mismatched model.
