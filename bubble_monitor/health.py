"""Data health checks for the daily alert: failed downloads and series that stopped updating."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Mapping

import pandas as pd

from .data_sources import cached, load_fred, load_yahoo

# Calendar days a series may lag before it counts as stale. Yahoo prices update every
# trading day (allow a long weekend); FRED credit/curve series publish with a 1-2 day lag.
STALE_DAYS = {"yahoo": 5, "fred": 8}


def _read_cache(cache_dir: Path | None, key: str) -> pd.Series | None:
    path = cache_dir / f"{key}.csv" if cache_dir is not None else None
    if path is None or not path.exists():
        return None
    try:
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    except Exception:
        return None


def load_with_health(fred_ids: Mapping[str, str], yahoo_ids: Mapping[str, str], cache_dir: Path | None,
                     max_age_seconds: float, today: dt.date | None = None) -> tuple[dict[str, pd.Series], list[str]]:
    """Download every series; return what loaded plus human-readable problems (never raises).

    When a download fails but an older cached copy exists, the old copy is used and the
    problem says how old it is, so one bad day at a data provider does not stop the alerts.
    """
    today = today or dt.date.today()
    raw: dict[str, pd.Series] = {}
    problems: list[str] = []
    sources = [(k, sid, "fred", load_fred) for k, sid in fred_ids.items()]
    sources += [(k, t, "yahoo", load_yahoo) for k, t in yahoo_ids.items()]
    for key, sid, kind, loader in sources:
        try:
            s = cached(cache_dir, key, lambda sid=sid, loader=loader: loader(sid), max_age_seconds=max_age_seconds)
        except Exception as exc:  # network, provider maintenance, delisted ticker...
            reason = str(exc)[:160]
            old = _read_cache(cache_dir, key)
            if old is not None and len(old.dropna()):
                problems.append(f"{sid}：下載失敗，改用 {old.dropna().index[-1]:%Y-%m-%d} 的舊資料（{reason}）")
                raw[key] = old.dropna()
            else:
                problems.append(f"{sid}：下載失敗，沒有舊資料可用（{reason}）")
            continue
        s = s.dropna()
        if not len(s):
            problems.append(f"{sid}：下載成功但沒有資料")
            continue
        raw[key] = s
        lag = (today - s.index[-1].date()).days
        if lag > STALE_DAYS[kind]:
            problems.append(f"{sid}：最新資料停在 {s.index[-1]:%Y-%m-%d}（已 {lag} 天沒更新）")
    return raw, problems
