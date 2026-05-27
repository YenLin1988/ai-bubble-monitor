"""SQLite-backed history storage for stress score observations."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


class HistoryStore:
    def __init__(self, db_path: str | Path, legacy_json_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path)
        self.legacy_json_path = Path(legacy_json_path) if legacy_json_path else None
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._migrate_legacy_json()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stress_history (
                    time TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alert_events (
                    alert_key TEXT PRIMARY KEY,
                    last_sent_time TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _migrate_legacy_json(self) -> None:
        if not self.legacy_json_path or not self.legacy_json_path.exists():
            return
        try:
            records = json.loads(self.legacy_json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(records, list):
            return
        for record in records:
            if isinstance(record, dict) and "time" in record:
                self.upsert_record(record)

    def upsert_record(self, record: dict[str, Any]) -> None:
        if "time" not in record:
            raise ValueError("history record requires a time field")
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO stress_history (time, payload) VALUES (?, ?)",
                (str(record["time"]), payload),
            )
            conn.commit()

    def load_history(self, limit: int = 2000) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT payload
                FROM stress_history
                ORDER BY time DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        records = [json.loads(row[0]) for row in reversed(rows)]
        return records

    def mark_alert_sent(self, alert_key: str, sent_time: str, payload: dict[str, Any]) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO alert_events (alert_key, last_sent_time, payload) VALUES (?, ?, ?)",
                (alert_key, sent_time, json.dumps(payload, ensure_ascii=False, sort_keys=True)),
            )
            conn.commit()

    def get_alert_event(self, alert_key: str) -> dict[str, Any] | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT last_sent_time, payload FROM alert_events WHERE alert_key = ?",
                (alert_key,),
            ).fetchone()
        if not row:
            return None
        return {"last_sent_time": row[0], "payload": json.loads(row[1])}
