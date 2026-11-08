"""Generate deterministic numerical test vectors, NOT a trained deployment model."""
import json
import sys
from pathlib import Path

import numpy as np

from swallow_ml.temporal import TCN

model = TCN(seed=7, hidden=3)
mean = np.arange(6) / 10
scale = np.arange(6) / 10 + 1
frames = np.sin(np.arange(96).reshape(16, 6) / 7)
export = model.export(mean, scale, .7, {"trained": False, "subjects": 0,
    "dataset_sha256": "0" * 64, "validation": {}, "test": {}, "purpose": "numerical_unit_test_only"})
fixture = {"model": export, "frames": frames.tolist(),
    "probability": float(model.predict(((frames - mean) / scale)[None])[0])}
Path(sys.argv[1]).write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
