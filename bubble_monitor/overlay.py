"""Risk-overlay strategies: turn warning / entry signals into an equity exposure and simulate it.

Rules (fixed before testing):
    warning overlay   go to ``defensive`` exposure when the market is within 10% of its
                      1-year high and the warning level reaches ``enter_level``; return to
                      full exposure on an entry signal, or after ``hold_days`` consecutive
                      days with warning level <= 1.
    Faber 10-month    100% when the month-end close is above its 10-month average, else cash
                      (Faber 2007), the standard academic trend benchmark.
    price-only        same state machine, but the warning needs only the market's own price:
                      below its 50-day average AND 20-day realised volatility >= 1.3x its
                      50-day average; entry = >= 15% below the 1-year high, realised vol
                      reached >= 30% within 20 days and has fallen >= 20% from that peak.

Execution: a signal seen at close t is traded at close t+1, so it affects returns
from t+1 onward. Uninvested money earns the T-bill rate. Each change in exposure
costs ``cost_per_trade`` of the traded fraction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def warning_overlay(sig: pd.DataFrame, enter_level: int = 4, defensive: float = 0.5,
                    hold_days: int = 5, entry_cols=("e_capitulation", "e_combo")) -> pd.Series:
    """Target exposure decided at each close."""
    level = sig["warning_level"].values
    near = sig["near_high"].values.astype(bool)
    entry = np.zeros(len(sig), dtype=bool)
    for c in entry_cols:
        if c in sig:
            entry |= sig[c].fillna(False).values.astype(bool)
    out = np.ones(len(sig))
    state, calm = 1.0, 0
    for i in range(len(sig)):
        calm = calm + 1 if level[i] <= 1 else 0
        if state == 1.0 and near[i] and level[i] >= enter_level:
            state = defensive
        elif state < 1.0 and (entry[i] or calm >= hold_days):
            state = 1.0
        out[i] = state
    return pd.Series(out, index=sig.index)


def price_only_signals(price: pd.Series) -> pd.DataFrame:
    """Warning level (0-2) and entry flag from a single price series."""
    p = price.dropna()
    rets = np.log(p).diff()
    rv = rets.rolling(20, min_periods=20).std() * np.sqrt(TRADING_DAYS)
    sma50 = p.rolling(50, min_periods=50).mean()
    dd_1y = p / p.rolling(TRADING_DAYS, min_periods=20).max() - 1
    trend = (p < sma50) & sma50.notna()
    vol = (rv >= 1.3 * rv.rolling(50, min_periods=50).mean()).fillna(False)
    rv_peak = rv.rolling(20, min_periods=1).max()
    entry = (dd_1y <= -0.15) & (rv_peak >= 0.30) & (rv <= 0.8 * rv_peak)
    return pd.DataFrame({
        "warning_level": trend.astype(int) + vol.astype(int),
        "near_high": dd_1y > -0.10,
        "e_entry": entry.fillna(False),
        "dd_1y": dd_1y,
    }, index=p.index)


def faber_exposure(price: pd.Series) -> pd.Series:
    """Faber (2007) 10-month SMA rule, decided at month-end closes, held through the next month."""
    p = price.dropna()
    m = p.resample("ME").last()
    above = (m > m.rolling(10, min_periods=10).mean()).astype(float)
    above[m.rolling(10, min_periods=10).mean().isna()] = 1.0
    # decision known at the last trading day of each month
    last_days = p.groupby(p.index.to_period("M")).apply(lambda s: s.index[-1])
    decided = pd.Series(above.values, index=pd.DatetimeIndex(last_days.values[:len(above)]))
    return decided.reindex(p.index).ffill().fillna(1.0)


def combined_exposure(trend: pd.Series, warning: pd.Series, entry: pd.Series | None = None,
                      entry_floor: float = 0.5) -> pd.Series:
    """Trend exposure (0/1) times warning exposure (0.5/1).

    With ``entry``: after an entry signal while the trend rule is out, hold at least
    ``entry_floor`` until the trend rule turns back in (addresses its slow re-entry).
    """
    idx = warning.index
    t = trend.reindex(idx).ffill().fillna(1.0).values
    w = warning.values
    e = entry.reindex(idx).fillna(False).astype(bool).values if entry is not None else np.zeros(len(idx), bool)
    out = np.empty(len(idx))
    floor_on = False
    for i in range(len(idx)):
        if t[i] >= 1:
            floor_on = False
        elif e[i]:
            floor_on = True
        base = t[i] * w[i]
        out[i] = max(base, entry_floor) if floor_on else base
    return pd.Series(out, index=idx)


def simulate(price: pd.Series, exposure: pd.Series, cash_rate: pd.Series | None = None,
             cost_per_trade: float = 0.001) -> pd.DataFrame:
    """Daily equity curve of holding ``exposure`` (decided at each close, traded next close)."""
    p = price.dropna()
    target = exposure.reindex(p.index).ffill().fillna(1.0)
    held = target.shift(2).fillna(1.0)  # decided at t, traded at t+1 close, earns from t+2
    eq_ret = p.pct_change().fillna(0.0)
    if cash_rate is not None:
        cr = cash_rate.reindex(p.index).ffill().fillna(0.0) / 100 / TRADING_DAYS
    else:
        cr = pd.Series(0.0, index=p.index)
    turnover = held.diff().abs().fillna(0.0)
    ret = held * eq_ret + (1 - held) * cr - turnover * cost_per_trade
    return pd.DataFrame({"ret": ret, "equity": (1 + ret).cumprod(), "exposure": held}, index=p.index)


def metrics(sim: pd.DataFrame, cash_rate: pd.Series | None = None) -> dict:
    eq = sim["equity"]
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = sim["ret"].std() * np.sqrt(TRADING_DAYS)
    mdd = (eq / eq.cummax() - 1).min()
    cr = 0.0
    if cash_rate is not None:
        cr = cash_rate.reindex(eq.index).ffill().fillna(0).mean() / 100
    return {
        "cagr": cagr, "vol": vol, "max_dd": mdd,
        "sharpe": (cagr - cr) / vol if vol else np.nan,
        "calmar": cagr / -mdd if mdd < 0 else np.nan,
        "time_defensive": (sim["exposure"] < 1).mean(),
        "trades": int((sim["exposure"].diff().abs() > 0).sum()),
    }


def bear_table(price: pd.Series, sims: dict[str, pd.DataFrame], events) -> pd.DataFrame:
    """Each strategy's loss over every benchmark bear market (peak to trough)."""
    rows = []
    for ev in events:
        row = {"peak": ev.peak_date.date(), "trough": ev.trough_date.date(), "index_drop": ev.depth}
        for name, sim in sims.items():
            e = sim["equity"]
            if ev.peak_date < e.index[0]:
                row[name] = np.nan
                continue
            row[name] = e.asof(ev.trough_date) / e.asof(ev.peak_date) - 1
        rows.append(row)
    return pd.DataFrame(rows)
