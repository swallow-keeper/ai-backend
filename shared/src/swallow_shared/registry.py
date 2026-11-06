from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from swallow_shared.embedding import EmbeddingExtractor, build_extractor
from swallow_shared.quality import QualityConfig


@dataclass(frozen=True)
class Registry:
    version: str
    embedding_backend: str
    efficientat_model: str
    efficientat_pretrained: bool
    wavlm_model: str
    tau1: float
    tau2: float
    k_consecutive: int
    ewma_alpha: float
    population_sigma: float
    baseline_min_passed: int
    baseline_max_spread: float
    profile_min_samples: int
    voice_timeout_seconds: float
    max_voice_attempts: int
    merge_window_seconds: float
    quality: QualityConfig

    def extractor(self) -> EmbeddingExtractor:
        return build_extractor(
            self.embedding_backend,
            efficientat_model=self.efficientat_model,
            efficientat_pretrained=self.efficientat_pretrained,
            wavlm_model=self.wavlm_model,
        )

    @property
    def model_version(self) -> str:
        return f"{self.embedding_backend}-{self.version}"


def load_registry(path: Path) -> Registry:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    embedding = raw["embedding"]
    score = raw["score"]
    baseline = raw["baseline"]
    voice = raw["voice"]
    quality = raw["quality"]
    return Registry(
        version=str(raw["version"]),
        embedding_backend=str(embedding["backend"]),
        efficientat_model=str(embedding["efficientat_model"]),
        efficientat_pretrained=bool(embedding["pretrained"]),
        wavlm_model=str(embedding["wavlm_model"]),
        tau1=float(score["tau1"]),
        tau2=float(score["tau2"]),
        k_consecutive=int(score["k_consecutive"]),
        ewma_alpha=float(score["ewma_alpha"]),
        population_sigma=float(score["population_sigma"]),
        baseline_min_passed=int(baseline["min_passed"]),
        baseline_max_spread=float(baseline["max_spread"]),
        profile_min_samples=int(baseline["profile_min_samples"]),
        voice_timeout_seconds=float(voice["timeout_seconds"]),
        max_voice_attempts=int(voice["max_attempts"]),
        merge_window_seconds=float(voice["merge_window_seconds"]),
        quality=QualityConfig(**quality),
    )
