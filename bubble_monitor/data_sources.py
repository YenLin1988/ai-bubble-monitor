"""Full-history downloads from FRED and Yahoo Finance with a small CSV cache."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd


def load_fred(series_id: str) -> pd.Series:
    df = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}", na_values=".").dropna()
    return pd.Series(df.iloc[:, 1].astype(float).values, index=pd.to_datetime(df.iloc[:, 0]))


def load_yahoo(ticker: str) -> pd.Series:
    import yfinance as yf

    hist = yf.Ticker(ticker).history(period="max", auto_adjust=False)
    if hist.empty or "Close" not in hist:
        raise ValueError(f"Yahoo returned no price history for {ticker}")
    close = hist["Close"].dropna()
    close.index = close.index.tz_localize(None).normalize()
    return close


def cached(cache_dir: Path | None, key: str, loader: Callable[[], pd.Series], refresh: bool = False,
           max_age_seconds: float = 86400) -> pd.Series:
    """Return ``loader()``, reusing ``cache_dir/key.csv`` while it is younger than ``max_age_seconds``."""
    if cache_dir is None:
        return loader()
    path = cache_dir / f"{key}.csv"
    if path.exists() and not refresh and time.time() - path.stat().st_mtime < max_age_seconds:
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    s = loader()
    cache_dir.mkdir(parents=True, exist_ok=True)
    s.to_frame(key).to_csv(path)
    return s


FINRA_MARGIN_URL = "https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx"


def load_finra_margin() -> pd.DataFrame:
    """FINRA monthly margin statistics (millions USD), indexed by month start.

    Columns: margin_debt (debit balances), free_credit (cash + margin account free credits).
    """
    import io
    import urllib.request

    req = urllib.request.Request(FINRA_MARGIN_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        x = pd.read_excel(io.BytesIO(resp.read()))
    idx = pd.to_datetime(x["Year-Month"].astype(str), format="%Y-%m")
    debit = pd.to_numeric(x.iloc[:, 1], errors="coerce").values
    free = (pd.to_numeric(x.iloc[:, 2], errors="coerce") + pd.to_numeric(x.iloc[:, 3], errors="coerce")).values
    return pd.DataFrame({"margin_debt": debit, "free_credit": free}, index=idx).sort_index()


def load_many(fred_ids: Mapping[str, str], yahoo_ids: Mapping[str, str], cache_dir: Path | None = None,
              refresh: bool = False, max_age_seconds: float = 86400) -> dict[str, pd.Series]:
    raw = {}
    for key, sid in fred_ids.items():
        raw[key] = cached(cache_dir, key, lambda sid=sid: load_fred(sid), refresh, max_age_seconds)
    for key, ticker in yahoo_ids.items():
        raw[key] = cached(cache_dir, key, lambda t=ticker: load_yahoo(t), refresh, max_age_seconds)
    return raw
