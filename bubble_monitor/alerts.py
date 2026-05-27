"""Alert rule evaluation for non-Telegram notification channels."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class Alert:
    key: str
    severity: str
    title: str
    message: str
    value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_alerts(context: dict[str, Any]) -> list[Alert]:
    alerts: list[Alert] = []
    stress_score = _num(context.get("stress_score"))
    delta_24h = _num(context.get("delta_24h"))
    nfci = _num(context.get("nfci"))
    m2_yoy = _num(context.get("m2_yoy"))
    consumer_sentiment = _num(context.get("consumer_sentiment"))
    buffett = _num(context.get("buffett_indicator"))
    regime = str(context.get("regime", ""))

    if stress_score is not None and stress_score >= 75:
        alerts.append(Alert("stress-danger", "critical", "Stress score entered danger zone", f"Weighted stress is {stress_score:.1f}/100.", stress_score))
    elif stress_score is not None and stress_score >= 65:
        alerts.append(Alert("stress-caution", "warning", "Stress score elevated", f"Weighted stress is {stress_score:.1f}/100.", stress_score))

    if delta_24h is not None and delta_24h >= 5:
        alerts.append(Alert("stress-velocity-24h", "warning", "Stress accelerated over 24H", f"24H stress velocity is {delta_24h:+.1f}.", delta_24h))

    if "泡沫破裂" in regime:
        alerts.append(Alert("regime-bust", "critical", "Regime changed to Bust", "Growth momentum is negative while macro stress is elevated."))

    if nfci is not None and nfci > 0.2:
        alerts.append(Alert("nfci-tightening", "warning", "Financial conditions tightening", f"NFCI is {nfci:+.2f}.", nfci))

    if m2_yoy is not None and m2_yoy < -2:
        alerts.append(Alert("m2-contraction", "critical", "M2 liquidity contraction", f"M2 YoY is {m2_yoy:+.1f}%.", m2_yoy))

    if consumer_sentiment is not None and consumer_sentiment < 55:
        alerts.append(Alert("consumer-sentiment-recession-watch", "warning", "Consumer sentiment recession watch", f"Consumer sentiment is {consumer_sentiment:.0f}.", consumer_sentiment))

    if buffett is not None and buffett > 200:
        alerts.append(Alert("buffett-extreme", "warning", "Buffett Indicator extreme valuation", f"Buffett Indicator is {buffett:.0f}%.", buffett))

    return alerts


def utc_now_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _num(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
