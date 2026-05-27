"""Data quality tracking for displayed market and macro series."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class DataPointStatus:
    name: str
    source: str
    status: str
    rows: int | None = None
    latest_date: str | None = None
    note: str = ""
    checked_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DataQualityReport:
    def __init__(self) -> None:
        self._items: list[DataPointStatus] = []

    def record(
        self,
        name: str,
        source: str,
        status: str,
        rows: int | None = None,
        latest_date: str | None = None,
        note: str = "",
    ) -> None:
        self._items.append(
            DataPointStatus(
                name=name,
                source=source,
                status=status,
                rows=rows,
                latest_date=latest_date,
                note=note,
                checked_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            )
        )

    def record_fred_frame(self, name: str, series_id: str, frame) -> None:
        if frame is None:
            self.record(name, f"FRED {series_id}", "error", note="No data returned")
            return
        rows = len(frame)
        latest_date = None
        try:
            latest_date = str(frame.iloc[-1, 0])
        except Exception:
            pass
        status = "fresh" if rows else "error"
        self.record(name, f"FRED {series_id}", status, rows=rows, latest_date=latest_date)

    def record_value(self, name: str, source: str, value, fallback_used: bool = False) -> None:
        if value is None:
            self.record(name, source, "fallback" if fallback_used else "error", note="Value unavailable")
        elif fallback_used:
            self.record(name, source, "fallback", note=f"Using fallback/default value {value}")
        else:
            self.record(name, source, "fresh")

    def rows(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._items]

    def counts(self) -> dict[str, int]:
        counts = {"fresh": 0, "fallback": 0, "stale": 0, "error": 0}
        for item in self._items:
            counts[item.status] = counts.get(item.status, 0) + 1
        return counts

    def health_score(self) -> int:
        if not self._items:
            return 0
        weights = {"fresh": 1.0, "stale": 0.7, "fallback": 0.4, "error": 0.0}
        score = sum(weights.get(item.status, 0.0) for item in self._items) / len(self._items)
        return round(score * 100)
