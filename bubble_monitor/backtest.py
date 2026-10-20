"""Historical backtest of the weighted stress score against S&P 500 drawdowns.

The live dashboard scores the latest value of each factor. This module rebuilds
the same factors month by month from long-history proxies, so the composite
score can be checked against past bear markets.

Look-ahead control: each month-end only sees data that would have been
published by then (see ``PUBLICATION_LAG_MONTHS`` and the quarterly lags in
``build_factor_values``). FRED serves revised data, not point-in-time
vintages, so revision bias remains (notably for NFCI and GDP).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .model_config import WEIGHTS
from .scoring import normalize

# Monthly macro releases that are published after the month they describe.
PUBLICATION_LAG_MONTHS = {"m2": 1, "unrate": 1}

# Z.1 equity market value is released ~10 weeks after quarter end; GDP advance ~4 weeks.
Z1_LAG = pd.DateOffset(months=3, days=70)
GDP_LAG = pd.DateOffset(months=3, days=30)

# FRED only serves the last 3 years of the ICE BofA HY OAS, so earlier history is
# approximated from Moody's Baa-10Y spread. Coefficients are a structural rule of
# thumb (2008: Baa 6.2 -> ~17, 2020: 4.3 -> ~11, calm: 1.6 -> ~3.3), not a fit.
HY_FROM_BAA_SLOPE = 3.0
HY_FROM_BAA_INTERCEPT = -1.5

# EV/EBITDA of AI mega-caps has no usable history; it is dropped and the
# remaining weights are renormalised each month.
UNAVAILABLE_FACTORS = ("EV/EBITDA",)


def _jpy_stress(row: pd.Series) -> float:
    stress = normalize(150 - row["usdjpy"], 0, 20)
    mom = row.get("jpy_mom")
    if mom is not None and not pd.isna(mom) and mom < -0.03:
        stress = min(100, stress + normalize(-mom, 0.03, 0.10) * 0.3)
    return stress


# Mirrors the risk_scores block in Ai_bubble_monitor.py. Keep the two in sync.
FACTOR_RULES: dict[str, tuple[tuple[str, ...], Callable[[pd.Series], float]]] = {
    "10Y Treasury": (("tnx",), lambda r: normalize(r["tnx"], 3.5, 5.0)),
    "Fed Funds Rate": (("fed_funds",), lambda r: normalize(r["fed_funds"], 3.0, 5.5)),
    "Net Liquidity": (("net_liquidity",), lambda r: normalize(6200 - r["net_liquidity"], 0, 800)),
    "M2 Supply YoY": (("m2_yoy",), lambda r: normalize(-r["m2_yoy"], -6, 2)),
    "JPY Liquidity": (("usdjpy",), _jpy_stress),
    "Yield Curve": (("curve",), lambda r: normalize(r["curve"], -0.8, 0.2)),
    "Credit Spread": (("spread",), lambda r: normalize(r["spread"], 3.0, 5.5)),
    "NFCI": (("nfci",), lambda r: normalize(r["nfci"], -0.5, 0.5)),
    "Buffett Indicator": (("buffett",), lambda r: normalize(r["buffett"], 100, 200)),
    "VXN Volatility": (("vxn",), lambda r: normalize(r["vxn"], 15, 35)),
    "Consumer Sentiment": (("umcsent",), lambda r: normalize(100 - r["umcsent"], 30, 70)),
    "US Dollar": (("dxy",), lambda r: normalize(r["dxy"], 100.0, 107.0)),
    "Commodities": (("comm_mom",), lambda r: normalize(r["comm_mom"], 0.0, 0.10)),
    "Unemployment": (("unrate",), lambda r: normalize(r["unrate"], 3.5, 6.0)),
}


@dataclass(frozen=True)
class DrawdownEvent:
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: pd.Timestamp | None
    depth: float  # negative, e.g. -0.34


# ---------------------------------------------------------------------------
# Factor history
# ---------------------------------------------------------------------------

def month_end_last(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Last observation on or before each month-end in ``index``."""
    s = series.dropna().sort_index()
    return s.reindex(s.index.union(index)).ffill().reindex(index)


def momentum_20d(daily: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    s = daily.dropna().sort_index()
    mom = s / s.shift(20) - 1
    # Only accept a momentum reading if the series actually traded near month-end.
    last_obs = pd.Series(s.index, index=s.index)
    stale = month_end_last(last_obs, index)
    out = month_end_last(mom, index)
    out[(index - pd.to_datetime(stale.values)) > pd.Timedelta(days=10)] = np.nan
    return out


def _available_from(series: pd.Series, lag: pd.DateOffset) -> pd.Series:
    """Re-index a quarterly FRED series (dated at quarter start) by publication date."""
    s = series.dropna()
    return pd.Series(s.values, index=s.index + lag).sort_index()


def build_factor_values(raw: Mapping[str, pd.Series], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Turn raw series into the month-end input values the factor rules expect.

    Required ``raw`` keys: gs10, fedfunds, walcl, wtregen, rrp, m2, usdjpy, t10y2y,
    hy, baa10y, nfci, corp_equity, gdp, vxn, vix, umcsent, dxy, oil, gold, unrate, spx.
    Optional (v2 model / longer warm-up): baa, oil_monthly, dollar_broad_old, rut, vix3m.
    """
    v = pd.DataFrame(index=index)

    v["tnx"] = month_end_last(raw["gs10"], index)
    v["fed_funds"] = month_end_last(raw["fedfunds"], index)

    walcl = month_end_last(raw["walcl"], index) / 1000
    tga = month_end_last(raw["wtregen"], index) / 1000
    rrp = month_end_last(raw["rrp"], index).fillna(0.0)
    v["net_liquidity"] = walcl - tga - rrp

    m2 = raw["m2"].dropna()
    m2_yoy = (m2 / m2.shift(12) - 1) * 100
    m2_yoy.index = m2_yoy.index + pd.DateOffset(months=PUBLICATION_LAG_MONTHS["m2"] + 1) - pd.Timedelta(days=1)
    v["m2_yoy"] = month_end_last(m2_yoy, index)

    v["usdjpy"] = month_end_last(raw["usdjpy"], index)
    v["jpy_mom"] = momentum_20d(raw["usdjpy"], index)
    v["curve"] = month_end_last(raw["t10y2y"], index)

    # Daily Baa-10Y starts in 1986; before that use monthly Baa minus GS10 so the
    # v2 trailing percentiles have a warm-up history.
    baa_spread = raw["baa10y"].dropna()
    baa_monthly = _get(raw, "baa").dropna()
    if len(baa_monthly):
        early = (baa_monthly - raw["gs10"].reindex(baa_monthly.index)).dropna()
        early.index = early.index + pd.DateOffset(months=1) - pd.Timedelta(days=1)
        cutoff = baa_spread.index.min() if len(baa_spread) else early.index.max() + pd.Timedelta(days=1)
        baa_spread = pd.concat([early[early.index < cutoff], baa_spread])
    proxy = baa_spread * HY_FROM_BAA_SLOPE + HY_FROM_BAA_INTERCEPT
    hy = raw["hy"].dropna()
    spread = pd.concat([proxy[proxy.index < hy.index.min()], hy]) if len(hy) else proxy
    v["spread"] = month_end_last(spread, index)

    v["nfci"] = month_end_last(raw["nfci"], index)

    # Buffett indicator: last published Z.1 equity value (millions) rolled forward
    # with the S&P 500, divided by last published GDP (billions SAAR).
    spx = raw["spx"].dropna()
    equity_q = raw["corp_equity"].dropna()
    spx_at_q_end = month_end_last(spx, pd.DatetimeIndex(equity_q.index + pd.DateOffset(months=3) - pd.Timedelta(days=1)))
    equity_scaled = pd.Series(equity_q.values / spx_at_q_end.values, index=equity_q.index)
    eq_per_spx = month_end_last(_available_from(equity_scaled, Z1_LAG), index)
    gdp = month_end_last(_available_from(raw["gdp"], GDP_LAG), index)
    v["buffett"] = eq_per_spx * month_end_last(spx, index) / 1000 / gdp * 100

    vxn = month_end_last(raw["vxn"], index)
    vix = month_end_last(raw["vix"], index)
    v["vxn"] = vxn.where(vxn.notna(), vix)

    v["umcsent"] = month_end_last(raw["umcsent"], index)
    v["dxy"] = month_end_last(raw["dxy"], index)
    v["comm_mom"] = pd.concat(
        [momentum_20d(raw["oil"], index), momentum_20d(raw["gold"], index)], axis=1
    ).mean(axis=1, skipna=True)

    unrate = raw["unrate"].dropna().copy()
    unrate.index = unrate.index + pd.DateOffset(months=PUBLICATION_LAG_MONTHS["unrate"] + 1) - pd.Timedelta(days=1)
    v["unrate"] = month_end_last(unrate, index)

    # Extra inputs used by the v2 model (bubble_monitor.model_v2).
    v["oil_m"] = month_end_last(_get(raw, "oil_monthly"), index)
    dxy_chg6 = v["dxy"] / v["dxy"].shift(6) - 1
    broad = month_end_last(_get(raw, "dollar_broad_old"), index)
    v["dxy_chg6"] = dxy_chg6.where(dxy_chg6.notna(), broad / broad.shift(6) - 1)
    v["rut_rel"] = month_end_last(_get(raw, "rut"), index) / month_end_last(spx, index)
    v["vix_ts"] = vix / month_end_last(_get(raw, "vix3m"), index)
    return v


def _get(raw: Mapping[str, pd.Series], key: str) -> pd.Series:
    s = raw.get(key)
    return s if s is not None else pd.Series(dtype=float, index=pd.DatetimeIndex([]))


def score_factors(values: pd.DataFrame) -> pd.DataFrame:
    scores = pd.DataFrame(index=values.index, columns=list(FACTOR_RULES), dtype=float)
    for name, (inputs, rule) in FACTOR_RULES.items():
        for date, row in values.iterrows():
            if all(pd.notna(row.get(col)) for col in inputs):
                scores.at[date, name] = rule(row)
    return scores


def composite_stress(scores: pd.DataFrame, weights: Mapping[str, float] = WEIGHTS) -> tuple[pd.Series, pd.Series]:
    """Weighted stress renormalised over the factors available each month.

    Returns (stress, coverage) where coverage is the share of total model weight
    that had data that month.
    """
    w = pd.Series({name: weights[name] for name in scores.columns})
    available = scores.notna()
    weight_present = available.mul(w, axis=1).sum(axis=1)
    weighted = scores.fillna(0).mul(w, axis=1).sum(axis=1)
    stress = weighted / weight_present.replace(0, np.nan)
    coverage = weight_present / sum(weights.values())
    return stress, coverage


# ---------------------------------------------------------------------------
# Market outcomes
# ---------------------------------------------------------------------------

def find_drawdown_events(prices: pd.Series, threshold: float = 0.20) -> list[DrawdownEvent]:
    """Peak-to-trough declines of at least ``threshold`` between all-time highs."""
    p = prices.dropna().sort_index()
    running_max = p.cummax()
    new_high = p >= running_max
    segment = new_high.cumsum()
    events = []
    for _, seg in p.groupby(segment):
        peak_date = seg.index[0]
        peak = seg.iloc[0]
        trough_date = seg.idxmin()
        depth = seg.loc[trough_date] / peak - 1
        if depth <= -threshold:
            after = p.loc[seg.index[-1]:]
            recovered = after[after >= peak]
            recovery = recovered.index[0] if len(recovered) and recovered.index[0] > seg.index[-1] else None
            events.append(DrawdownEvent(peak_date, trough_date, recovery, float(depth)))
    return events


def forward_max_drawdown(prices: pd.Series, index: pd.DatetimeIndex, months: int = 12) -> pd.Series:
    """Worst peak-to-trough decline within the next ``months`` after each date.

    The starting price counts as the initial peak. NaN when the window is incomplete.
    """
    p = prices.dropna().sort_index()
    last = p.index[-1]
    out = pd.Series(np.nan, index=index)
    for date in index:
        end = date + pd.DateOffset(months=months)
        if end > last:
            continue
        window = p.loc[(p.index >= date) & (p.index <= end)]
        if len(window) < 2:
            continue
        out[date] = float((window / window.cummax() - 1).min())
    return out


def forward_return(prices: pd.Series, index: pd.DatetimeIndex, months: int) -> pd.Series:
    start = month_end_last(prices, index)
    end_idx = pd.DatetimeIndex(index + pd.DateOffset(months=months))
    end = month_end_last(prices, end_idx)
    end[end_idx > prices.dropna().index[-1]] = np.nan
    return pd.Series(end.values / start.values - 1, index=index)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def event_table(stress: pd.Series, events: list[DrawdownEvent], thresholds=(50, 65, 75), lookback_months: int = 12) -> pd.DataFrame:
    rows = []
    s = stress.dropna()
    for ev in events:
        peak_month = ev.peak_date + pd.offsets.MonthEnd(0)
        window = s.loc[peak_month - pd.DateOffset(months=lookback_months):peak_month]
        row = {
            "peak": ev.peak_date.date(),
            "trough": ev.trough_date.date(),
            "depth": ev.depth,
            "score_at_peak": s.get(peak_month, np.nan),
            "score_6m_before": _asof(s, peak_month - pd.DateOffset(months=6)),
            "score_12m_before": _asof(s, peak_month - pd.DateOffset(months=12)),
            "max_score_12m_before": window.max() if len(window) else np.nan,
        }
        for t in thresholds:
            hits = window[window >= t]
            row[f"lead_months_{t}"] = (
                (peak_month.year - hits.index[0].year) * 12 + peak_month.month - hits.index[0].month
                if len(hits) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _asof(s: pd.Series, date) -> float:
    sub = s.loc[:date]
    return float(sub.iloc[-1]) if len(sub) else np.nan


def threshold_table(
    stress: pd.Series,
    fwd_dd: pd.Series,
    events: list[DrawdownEvent],
    thresholds=range(30, 85, 5),
    crash_depth: float = 0.20,
    lookback_months: int = 12,
    near_high: pd.Series | None = None,
) -> pd.DataFrame:
    """Signal quality of ``stress >= threshold`` as a crash warning.

    precision: share of alert months followed by a >= crash_depth drawdown within 12m.
    precision_near_high: same, restricted to months flagged in ``near_high`` (market
    still close to its high), which excludes alerts raised mid-crash.
    event_recall: share of crash events with at least one alert in the 12m before the peak.
    """
    df = pd.DataFrame({"stress": stress, "fwd_dd": fwd_dd}).dropna()
    crash_ahead = df["fwd_dd"] <= -crash_depth
    base_rate = crash_ahead.mean()
    nh = near_high.reindex(df.index).fillna(False).astype(bool) if near_high is not None else None
    base_rate_nh = crash_ahead[nh].mean() if nh is not None else np.nan
    s = stress.dropna()
    rows = []
    for t in thresholds:
        alert = df["stress"] >= t
        precision = crash_ahead[alert].mean() if alert.any() else np.nan
        caught = 0
        for ev in events:
            pm = ev.peak_date + pd.offsets.MonthEnd(0)
            window = s.loc[pm - pd.DateOffset(months=lookback_months):pm]
            caught += bool((window >= t).any())
        nh_alert = alert & nh if nh is not None else None
        precision_nh = crash_ahead[nh_alert].mean() if nh_alert is not None and nh_alert.any() else np.nan
        rows.append({
            "threshold": t,
            "time_in_alert": (s >= t).mean(),
            "precision": precision,
            "lift": precision / base_rate if base_rate and not pd.isna(precision) else np.nan,
            "precision_near_high": precision_nh,
            "lift_near_high": precision_nh / base_rate_nh if base_rate_nh and not pd.isna(precision_nh) else np.nan,
            "event_recall": caught / len(events) if events else np.nan,
            "events_caught": f"{caught}/{len(events)}",
        })
    out = pd.DataFrame(rows)
    out.attrs["base_rate"] = base_rate
    out.attrs["base_rate_near_high"] = base_rate_nh
    return out


def bucket_table(stress: pd.Series, fwd_ret_12m: pd.Series, fwd_dd: pd.Series, bins=(0, 25, 50, 65, 75, 101)) -> pd.DataFrame:
    df = pd.DataFrame({"stress": stress, "ret": fwd_ret_12m, "dd": fwd_dd}).dropna()
    df["bucket"] = pd.cut(df["stress"], bins=bins, right=False)
    g = df.groupby("bucket", observed=False)
    return pd.DataFrame({
        "months": g.size(),
        "avg_fwd_12m_return": g["ret"].mean(),
        "avg_fwd_12m_max_dd": g["dd"].mean(),
        "p_dd_10": g["dd"].apply(lambda x: (x <= -0.10).mean() if len(x) else np.nan),
        "p_dd_20": g["dd"].apply(lambda x: (x <= -0.20).mean() if len(x) else np.nan),
    })


def factor_diagnostics(scores: pd.DataFrame, fwd_dd: pd.Series, events: list[DrawdownEvent], lookback_months: int = 12) -> pd.DataFrame:
    """Per-factor behaviour: saturation, pre-crash level vs normal, and correlation with forward drawdown."""
    pre_crash = pd.Series(False, index=scores.index)
    for ev in events:
        pm = ev.peak_date + pd.offsets.MonthEnd(0)
        pre_crash[(scores.index >= pm - pd.DateOffset(months=lookback_months)) & (scores.index <= pm)] = True
    rows = []
    for name in scores.columns:
        col = scores[name]
        valid = col.dropna()
        joined = pd.concat([col, fwd_dd], axis=1).dropna()
        rows.append({
            "factor": name,
            "first_month": valid.index.min().date() if len(valid) else None,
            "pct_at_100": (valid >= 99.9).mean() if len(valid) else np.nan,
            "pct_at_0": (valid <= 0.1).mean() if len(valid) else np.nan,
            "avg_pre_crash": col[pre_crash].mean(),
            "avg_other": col[~pre_crash].mean(),
            # Positive = higher score precedes deeper drawdowns (fwd_dd is negative).
            "corr_with_fwd_dd": -joined.iloc[:, 0].rank().corr(joined.iloc[:, 1].rank()) if len(joined) > 24 else np.nan,
        })
    return pd.DataFrame(rows)


def auc(score: pd.Series, label: pd.Series) -> float:
    """Probability a random crash-ahead month scores higher than a random other month."""
    df = pd.concat([score, label], axis=1).dropna()
    pos, neg = df.iloc[:, 1].astype(bool), ~df.iloc[:, 1].astype(bool)
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = df.iloc[:, 0].rank()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def model_summary(
    stress: pd.Series,
    fwd_dd: pd.Series,
    events: list[DrawdownEvent],
    near_high: pd.Series,
    crash_depth: float,
    periods: Mapping[str, tuple[str, str]],
    alert_quantile: float = 0.80,
    lookback_months: int = 12,
) -> dict[str, object]:
    """Threshold-free and fixed-alert-rate comparison metrics for one model.

    The alert line is the model's own ``alert_quantile`` over the sample, so every
    model is judged with the same share of time in alert.
    """
    crash_ahead = (fwd_dd <= -crash_depth).where(fwd_dd.notna())
    nh = near_high.reindex(stress.index).fillna(False).astype(bool)
    out: dict[str, object] = {
        "auc_all": auc(stress, crash_ahead),
        "auc_near_high": auc(stress[nh], crash_ahead[nh]),
    }
    for label, (start, end) in periods.items():
        mask = nh & (stress.index >= start) & (stress.index <= end)
        out[f"auc_near_high_{label}"] = auc(stress[mask], crash_ahead[mask])
    line = float(stress.quantile(alert_quantile))
    alert = stress >= line
    hit = crash_ahead[alert & nh].dropna()
    out["alert_line"] = line
    out["precision_near_high_at_line"] = float(hit.mean()) if len(hit) else np.nan
    caught = []
    s = stress.dropna()
    for ev in events:
        pm = ev.peak_date + pd.offsets.MonthEnd(0)
        caught.append(bool((s.loc[pm - pd.DateOffset(months=lookback_months):pm] >= line).any()))
    out["events_caught_at_line"] = f"{sum(caught)}/{len(caught)}"
    return out
