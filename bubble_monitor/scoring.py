"""Pure scoring helpers for regime classification and stress decomposition."""

from __future__ import annotations

from dataclasses import dataclass
import datetime
from typing import Mapping

from .model_config import RISK_CLUSTERS, WEIGHTS


@dataclass(frozen=True)
class Regime:
    name: str
    color: str
    css: str
    icon: str


def normalize(value: float, safe_val: float, danger_val: float) -> float:
    """Linearly map a value onto a 0-100 stress score."""
    if danger_val == safe_val:
        raise ValueError("safe_val and danger_val must differ")
    score = ((value - safe_val) / (danger_val - safe_val)) * 100
    return min(100, max(0, score))


def weighted_stress_score(
    risk_scores: Mapping[str, float],
    weights: Mapping[str, float] = WEIGHTS,
) -> float:
    missing = set(risk_scores) - set(weights)
    if missing:
        raise KeyError(f"Missing weights for factors: {sorted(missing)}")
    return sum(risk_scores[name] * weights[name] for name in risk_scores)


def classify_regime(growth_score: float, stress_score: float) -> Regime:
    if growth_score >= 0 and stress_score < 50:
        return Regime("擴張期 Expansion", "#2EA043", "safe", "●")
    if growth_score >= 0 and stress_score >= 50:
        return Regime("過熱期 Overheating", "#D29922", "caution", "▲")
    if growth_score < 0 and stress_score >= 50:
        return Regime("泡沫破裂 Bust", "#F85149", "danger", "◆")
    return Regime("沉澱期 Contraction", "#58A6FF", "cool", "■")


def cluster_decomposition(
    risk_scores: Mapping[str, float],
    weights: Mapping[str, float] = WEIGHTS,
    clusters: Mapping[str, str] = RISK_CLUSTERS,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, score in risk_scores.items():
        cluster = clusters[name]
        if cluster not in result:
            result[cluster] = {"total": 0.0, "weight_sum": 0.0, "factors": []}
        result[cluster]["total"] = float(result[cluster]["total"]) + score * weights[name]
        result[cluster]["weight_sum"] = float(result[cluster]["weight_sum"]) + weights[name]
        factors = result[cluster]["factors"]
        assert isinstance(factors, list)
        factors.append((name, score, weights[name]))
    return result


def stress_delta(history: list[dict[str, object]], current_score: float, hours_back: int, now) -> float | None:
    if len(history) < 2:
        return None
    cutoff = now - datetime.timedelta(hours=hours_back)
    past_records = []
    for record in history[:-1]:
        try:
            record_time = datetime.datetime.strptime(str(record["time"]), "%Y-%m-%d %H:%M")
        except (KeyError, ValueError, TypeError):
            continue
        if record_time <= cutoff:
            past_records.append(record)
    ref = past_records[-1] if past_records else history[0]
    try:
        return current_score - float(ref["stress_score"])
    except (KeyError, TypeError, ValueError):
        return None
