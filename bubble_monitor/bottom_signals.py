"""Candidate market-bottom signals and an honest comparison harness.

Every candidate uses either its published parameters or a rule fixed before
testing; nothing here is tuned to the backtest. Candidates are compared on two
separate periods so a rule has to work in both, not just on the full sample.

Not testable with free data (and therefore absent): Zweig Breadth Thrust and
Lowry 90% days need NYSE advance/decline and up/down volume history.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .daily_signals import TRADING_DAYS_YEAR, episode_starts


def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return 100 - 100 / (1 + gain / loss)


def coppock(monthly_close: pd.Series) -> pd.Series:
    """Coppock (1962): 10-month WMA of the 14-month plus 11-month rate of change."""
    roc = (monthly_close / monthly_close.shift(14) - 1) + (monthly_close / monthly_close.shift(11) - 1)
    w = np.arange(1, 11)
    return roc.rolling(10).apply(lambda a: np.dot(a, w) / w.sum(), raw=True)


def _to_daily(flags: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    """Place a lower-frequency signal on the first trading day on/after its period end."""
    out = pd.Series(False, index=daily_index)
    for dt in flags.index[flags.fillna(False).astype(bool)]:
        pos = daily_index.searchsorted(dt)
        if pos < len(daily_index):
            out.iloc[pos] = True
    return out


def sig_coppock(d: pd.DataFrame, s: pd.DataFrame) -> pd.Series:
    m = d["spx"].resample("ME").last()
    c = coppock(m)
    return _to_daily((c < 0) & (c > c.shift(1)) & (c.shift(1) <= c.shift(2)), d.index)


def sig_weekly_rsi(d: pd.DataFrame, s: pd.DataFrame) -> pd.Series:
    w = d["spx"].resample("W-FRI").last()
    rsi = wilder_rsi(w)
    return _to_daily((rsi > 30) & (rsi.shift(1) <= 30), d.index)


def sig_capitulation(d: pd.DataFrame, s: pd.DataFrame) -> pd.Series:
    return s["e_capitulation"]


def sig_credit_peak(d: pd.DataFrame, s: pd.DataFrame) -> pd.Series:
    return s["e_credit_peak"]


def sig_vix_ts_flip(d: pd.DataFrame, s: pd.DataFrame) -> pd.Series:
    return s["e_vix_ts_flip"]


def sig_combo(d: pd.DataFrame, s: pd.DataFrame) -> pd.Series:
    return s["e_combo"]


CANDIDATES: dict[str, tuple[str, str, Callable[[pd.DataFrame, pd.DataFrame], pd.Series]]] = {
    "capitulation": ("恐慌高峰（現行）", "跌 ≥15%、VIX 曾 ≥35 且已從高點回落 ≥20%", sig_capitulation),
    "coppock": ("Coppock 曲線", "月線 14+11 個月變動率的 10 月加權均線，在 0 以下轉折向上（1962 原始參數）", sig_coppock),
    "weekly_rsi": ("週 RSI 回升", "週 RSI(14) 從 30 以下回到 30 以上（Wilder 標準參數）", sig_weekly_rsi),
    "credit_peak": ("信用利差停止擴大", "跌 ≥15%，40 天內利差 20 日擴大曾 ≥1 個百分點，現在 20 日變動轉為 ≤0", sig_credit_peak),
    "vix_ts_flip": ("VIX 期限結構轉正", "跌 ≥10%，VIX/VIX3M 從 ≥1（倒掛）回到 <1", sig_vix_ts_flip),
    "combo": ("組合（≥2 個訊號）", "恐慌高峰、信用利差停止擴大、VIX 期限結構轉正，20 天內至少 2 個", sig_combo),
}


def decline_troughs(spx: pd.Series, dd_1y: pd.Series, depth: float = 0.15) -> pd.DataFrame:
    """Each decline of >= ``depth`` below the 1-year high, with its lowest close."""
    rows = []
    crossings = episode_starts(dd_1y <= -depth, gap_days=TRADING_DAYS_YEAR // 2)
    for cross in crossings:
        after = dd_1y.loc[cross:]
        rec = after[after > -0.05]
        end = rec.index[0] if len(rec) else after.index[-1]
        seg = spx.loc[cross:end]
        rows.append({"crossed": cross, "trough": seg.idxmin(), "depth": float(dd_1y.loc[cross:end].min())})
    return pd.DataFrame(rows)


def evaluate_candidate(signal: pd.Series, spx: pd.Series, fwd: pd.DataFrame, troughs: pd.DataFrame,
                       near_days: int = 30) -> tuple[dict, pd.DataFrame]:
    """Summary stats and per-signal rows.

    near_trough: share of signals within ``near_days`` trading days of a decline's true low.
    troughs_caught: declines with a signal within ``near_days`` of their low.
    """
    dates = episode_starts(signal.fillna(False).astype(bool))
    idx = spx.index
    trough_pos = [idx.get_loc(t) for t in troughs["trough"]] if len(troughs) else []
    rows = []
    for dt in dates:
        i = idx.get_loc(dt)
        dist = min((i - tp for tp in trough_pos), key=abs) if trough_pos else np.nan
        rows.append({
            "date": dt, "days_from_low": dist,
            "ret_63": fwd.at[dt, "ret_63"], "ret_252": fwd.at[dt, "ret_252"],
            "further_drop_126": fwd.at[dt, "lowest_126"],
        })
    eps = pd.DataFrame(rows)
    caught = 0
    sig_pos = [idx.get_loc(dt) for dt in dates]
    for tp in trough_pos:
        caught += any(abs(sp - tp) <= near_days for sp in sig_pos)
    done = eps.dropna(subset=["ret_252"]) if len(eps) else eps
    summary = {
        "signals": len(eps),
        "win_252": (done["ret_252"] > 0).mean() if len(done) else np.nan,
        "avg_252": done["ret_252"].mean() if len(done) else np.nan,
        "median_252": done["ret_252"].median() if len(done) else np.nan,
        "worst_252": done["ret_252"].min() if len(done) else np.nan,
        "avg_further_drop": eps["further_drop_126"].mean() if len(eps) else np.nan,
        "worst_further_drop": eps["further_drop_126"].min() if len(eps) else np.nan,
        "near_trough": (eps["days_from_low"].abs() <= near_days).mean() if len(eps) else np.nan,
        "troughs_caught": f"{caught}/{len(trough_pos)}",
    }
    return summary, eps
