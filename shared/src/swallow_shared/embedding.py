from __future__ import annotations

import contextlib
import io
from typing import Protocol

import numpy as np

from swallow_shared.constants import SAMPLE_RATE
from swallow_shared.mel import power_mel


class EmbeddingExtractor(Protocol):
    name: str

    def embed(self, samples: np.ndarray) -> np.ndarray: ...


class EmbeddingUnavailable(Exception):
    def __init__(self, backend: str):
        self.backend = backend
        super().__init__(backend)


class MelPoolExtractor:
    name = "mel_pool"

    def embed(self, samples: np.ndarray) -> np.ndarray:
        mel = np.log(power_mel(samples) + 1e-10)
        vector = np.concatenate([mel.mean(axis=1), mel.std(axis=1)]).astype(np.float32)
        return _normalize(vector)


_WIDTH = {
    "mn10_as": 1.0,
    "mn30_as": 3.0,
    "mn40_as": 4.0,
    "mn30": 3.0,
    "mn40": 4.0,
}


class EfficientATExtractor:
    # 공용 log-mel을 넣고, EfficientAT 내부 mel은 쓰지 않음
    name = "efficientat"

    def __init__(self, model_name: str, pretrained: bool):
        self.model_name = model_name
        self.pretrained = pretrained
        self._model = None

    def embed(self, samples: np.ndarray) -> np.ndarray:
        model, torch = self._load()
        spec = np.log(power_mel(samples) + 1e-10).astype(np.float32)
        tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            _preds, features = model(tensor)
        return _normalize(features.detach().cpu().numpy().reshape(-1))

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            from models.mn.model import get_model
        except ImportError as exc:
            raise EmbeddingUnavailable(self.name) from exc
        width = _WIDTH.get(self.model_name, 1.0)
        pretrained_name = self.model_name if self.pretrained else None
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            model = get_model(
                pretrained_name=pretrained_name,
                width_mult=width,
                head_type="mlp",
                input_dim_f=128,
                input_dim_t=200,
            )
        model.eval()
        self._model = (model, torch)
        return self._model


class WavLMExtractor:
    name = "wavlm"

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def embed(self, samples: np.ndarray) -> np.ndarray:
        model, torch = self._load()
        waveform = _resample(samples, SAMPLE_RATE, 16_000)
        tensor = torch.from_numpy(waveform).unsqueeze(0)
        with torch.no_grad():
            hidden = model(tensor).last_hidden_state
        pooled = hidden.mean(dim=1).squeeze(0).detach().cpu().numpy()
        return _normalize(pooled)

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            from transformers import WavLMModel
        except ImportError as exc:
            raise EmbeddingUnavailable(self.name) from exc
        model = WavLMModel.from_pretrained(self.model_name)
        model.eval()
        self._model = (model, torch)
        return self._model


class ClassifierHead:
    # 임상 라벨 전 비활성, 호출하지 않음
    active = False

    def predict(self, embedding: np.ndarray) -> np.ndarray:
        raise RuntimeError("분류 head 비활성")


def build_extractor(
    backend: str,
    *,
    efficientat_model: str,
    efficientat_pretrained: bool,
    wavlm_model: str,
) -> EmbeddingExtractor:
    if backend == "mel_pool":
        return MelPoolExtractor()
    if backend == "efficientat":
        return EfficientATExtractor(efficientat_model, efficientat_pretrained)
    if backend == "wavlm":
        return WavLMExtractor(wavlm_model)
    raise ValueError(backend)


def _normalize(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    return array / (np.linalg.norm(array) + 1e-8)


def _resample(samples: np.ndarray, source: int, target: int) -> np.ndarray:
    if source == target:
        return samples.astype(np.float32, copy=False)
    from math import gcd

    from scipy.signal import resample_poly

    divisor = gcd(source, target)
    converted = resample_poly(samples, target // divisor, source // divisor)
    return np.ascontiguousarray(converted, dtype=np.float32)
