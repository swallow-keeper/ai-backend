from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Decision:
    state: str
    mild_streak: int
    band: str


def change_score(embedding: np.ndarray, baseline: np.ndarray, sigma: float) -> tuple[float, float]:
    # 변화 점수, 기준 중앙값과의 거리 / 개인 변동 폭
    distance = float(np.linalg.norm(np.asarray(embedding) - np.asarray(baseline)))
    return distance / max(float(sigma), 1e-6), distance


def ewma(previous: float | None, value: float, alpha: float) -> float:
    if previous is None:
        return float(value)
    return float(alpha * value + (1.0 - alpha) * previous)


def decide(
    score: float,
    *,
    in_recheck: bool,
    mild_streak: int,
    tau1: float,
    tau2: float,
    k_consecutive: int,
) -> Decision:
    if score >= tau2:
        if in_recheck:
            return Decision("caution", 0, "high")
        return Decision("recheck", 0, "high")
    if score >= tau1:
        streak = mild_streak + 1
        if streak >= k_consecutive:
            return Decision("recheck", 0, "mild")
        return Decision("no_change", streak, "mild")
    return Decision("no_change", 0, "low")


def median_and_spread(vectors: list[np.ndarray]) -> tuple[np.ndarray, float]:
    stack = np.stack([np.asarray(vector, dtype=np.float32) for vector in vectors])
    median = np.median(stack, axis=0).astype(np.float32)
    distances = np.linalg.norm(stack - median, axis=1)
    return median, float(np.std(distances))


def baseline_reasons(passed_count: int, spread: float, min_passed: int, max_spread: float) -> list[str]:
    reasons: list[str] = []
    if passed_count < min_passed:
        reasons.append("BASELINE_TOO_FEW")
    if passed_count >= 2 and spread > max_spread:
        reasons.append("BASELINE_UNSTABLE")
    return reasons


def sigma_for_user(std: float | None, count: int, population_sigma: float, min_samples: int) -> float:
    if std is None or count < min_samples or std < 1e-6:
        return population_sigma
    return float(std)
