"""Daily early-warning and entry signals for the S&P 500.

Two independent signal families, evaluated on every trading day:

Early warning (risk rising while the market is still near its high)
    Five yes/no conditions; the warning level is how many are true.
    credit   HY spread widened >= 0.40pp over 20 trading days
    breadth  Russell 2000 lagged the S&P 500 by >= 5% over 60 trading days
    vol      VIX / VIX3M >= 0.95 (term structure flattening toward inversion);
             before VIX3M exists: VIX >= 1.3x its 50-day average
    trend    S&P 500 closed below its 50-day moving average
    curve    10Y-2Y is positive again after an inversion within ~2 years
    Levels: 0-1 normal, 2 caution, 3+ warning. Only meaningful while the market
    is less than 10% below its 1-year high; deeper than that it is already a
    correction and the entry signals take over.

Entry (the selling looks exhausted after a real decline)
    capitulation  >= 15% below the 1-year high, VIX hit >= 35 within 20 days
                  and has since fallen >= 20% from that spike (fear receding)
    combo         at least two of {capitulation, credit stress fading, VIX term
                  structure back to contango} within 20 days

These rules were fixed before looking at their backtest results and are not
tuned to any specific crash. They report historical odds, not certainties.
The combo replaced an earlier "back above the 50-day average" confirmation
signal that tested no better than buying at the first -15%; see
scripts/run_bottom_study.py for the candidate comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .backtest import HY_FROM_BAA_INTERCEPT, HY_FROM_BAA_SLOPE

TRADING_DAYS_YEAR = 252
CORRECTION_DD = -0.10
HOLD_DAYS = 5
STATS_START = "1990-01-01"  # first year with every warning input (VIX) available

WARNING_CONDITIONS = {
    "credit": "信用利差 20 日擴大 ≥0.40 個百分點",
    "breadth": "小型股 60 日落後大盤 ≥5%",
    "vol": "VIX 期限結構逼近倒掛（VIX/VIX3M ≥0.95）",
    "trend": "指數跌破 50 日均線",
    "curve": "殖利率曲線倒掛後重新轉正",
}
ENTRY_SIGNALS = {
    "capitulation": "恐慌高峰：跌 ≥15%、VIX 曾 ≥35 且已從高點回落 ≥20%",
    "combo": "多重確認：恐慌高峰、信用利差停止擴大、VIX 期限結構轉正，20 天內出現至少 2 個",
}
WARNING_LEVEL_NAMES = {0: "正常", 1: "正常", 2: "注意", 3: "警告", 4: "警告", 5: "警告"}


def hy_spread_series(hy: pd.Series, baa10y: pd.Series) -> pd.Series:
    """ICE HY OAS where FRED has it, earlier history approximated from Baa-10Y."""
    hy = hy.dropna()
    proxy = baa10y.dropna() * HY_FROM_BAA_SLOPE + HY_FROM_BAA_INTERCEPT
    if not len(hy):
        return proxy
    return pd.concat([proxy[proxy.index < hy.index.min()], hy]).sort_index()


def build_daily_frame(raw: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Align inputs on S&P 500 trading days. FRED series are forward-filled up to 5 days
    because they publish with a one-day lag."""
    spx = raw["spx"].dropna().sort_index()
    idx = spx.index

    def align(s: pd.Series | None, limit: int = 5) -> pd.Series:
        if s is None or not len(s.dropna()):
            return pd.Series(np.nan, index=idx)
        s = s.dropna().sort_index()
        s = s[~s.index.duplicated(keep="last")]
        return s.reindex(s.index.union(idx)).ffill(limit=limit).reindex(idx)

    return pd.DataFrame({
        "spx": spx,
        "spread": align(hy_spread_series(raw.get("hy", pd.Series(dtype=float)), raw["baa10y"])),
        "rut": align(raw.get("rut"), limit=2),
        "vix": align(raw.get("vix_daily"), limit=2),
        "vix3m": align(raw.get("vix3m_daily"), limit=2),
        "curve": align(raw.get("t10y2y")),
    }, index=idx)


def compute_signals(d: pd.DataFrame, target: pd.Series | None = None) -> pd.DataFrame:
    """Warning and entry signals for ``target`` (default: the S&P 500 in ``d``).

    Index-specific parts (distance from high, 50-day trend, decline thresholds) use
    ``target``; market-wide inputs (credit, small-cap breadth vs the S&P 500, VIX,
    yield curve) are the same whichever index is targeted.
    """
    out = pd.DataFrame(index=d.index)
    spx = d["spx"]
    px = spx if target is None else target.reindex(d.index).ffill(limit=2)
    high_1y = px.rolling(TRADING_DAYS_YEAR, min_periods=20).max()
    out["dd_1y"] = px / high_1y - 1
    out["dd_ath"] = px / px.cummax() - 1
    out["vix"] = d["vix"]
    sma50 = px.rolling(50, min_periods=50).mean()

    # --- early warning ---
    spread_chg = d["spread"] - d["spread"].shift(20)
    rel = d["rut"] / spx
    rel_chg = rel / rel.shift(60) - 1
    ts = d["vix"] / d["vix3m"]
    vix_vs_avg = d["vix"] / d["vix"].rolling(50, min_periods=50).mean()
    curve_min = d["curve"].rolling(2 * TRADING_DAYS_YEAR, min_periods=1).min()

    cond = pd.DataFrame(index=d.index)
    cond["credit"] = spread_chg >= 0.40
    cond["breadth"] = rel_chg <= -0.05
    cond["vol"] = np.where(ts.notna(), ts >= 0.95, vix_vs_avg >= 1.3)
    cond["trend"] = px < sma50
    cond["curve"] = (d["curve"] > 0) & (curve_min < 0)
    # Unknown inputs count as "not triggered" but are tracked in coverage.
    known = pd.DataFrame({
        "credit": spread_chg.notna(), "breadth": rel_chg.notna(),
        "vol": ts.notna() | vix_vs_avg.notna(), "trend": sma50.notna(), "curve": d["curve"].notna(),
    })
    cond = cond.astype(bool) & known
    for c in WARNING_CONDITIONS:
        out[f"w_{c}"] = cond[c]
    out["warning_level"] = cond.sum(axis=1)
    out["warning_inputs"] = known.sum(axis=1)
    out["near_high"] = out["dd_1y"] > CORRECTION_DD

    # --- entry ---
    vix_peak20 = d["vix"].rolling(20, min_periods=1).max()
    out["e_capitulation"] = (out["dd_1y"] <= -0.15) & (vix_peak20 >= 35) & (d["vix"] <= 0.8 * vix_peak20)
    # Credit stress fading: spreads had widened >= 1pp in 20 days, now no longer widening.
    stressed = spread_chg.rolling(40, min_periods=1).max() >= 1.0
    out["e_credit_peak"] = (out["dd_1y"] <= -0.15) & stressed & (spread_chg <= 0) & (spread_chg.shift(1) > 0)
    # VIX term structure flips from backwardation back to contango.
    out["e_vix_ts_flip"] = (out["dd_1y"] <= -0.10) & (ts < 1.0) & (ts.shift(1) >= 1.0)
    # Combo: one of the three fires today and at least two fired within 20 days.
    parts = out[["e_capitulation", "e_credit_peak", "e_vix_ts_flip"]].fillna(False).astype(int)
    out["e_combo"] = (parts.rolling(20, min_periods=1).max().sum(axis=1) >= 2) & (parts.sum(axis=1) > 0)
    return out


# ---------------------------------------------------------------------------
# Historical evaluation
# ---------------------------------------------------------------------------

def episode_starts(mask: pd.Series, gap_days: int = 60) -> pd.DatetimeIndex:
    """First day of each run of True, merging runs closer than ``gap_days`` trading days."""
    m = mask.fillna(False).astype(bool).values
    starts, last_true = [], -10**9
    for i, v in enumerate(m):
        if v:
            if i - last_true > gap_days:
                starts.append(i)
            last_true = i
    return mask.index[starts]


def forward_outcomes(spx: pd.Series, horizons=(63, 126, 252)) -> pd.DataFrame:
    """Forward return and forward max drawdown (from the starting price) per horizon in trading days."""
    p = spx.values.astype(float)
    n = len(p)
    out = {}
    for h in horizons:
        ret = np.full(n, np.nan)
        mdd = np.full(n, np.nan)
        low = np.full(n, np.nan)
        for i in range(n - h):
            w = p[i:i + h + 1]
            ret[i] = w[-1] / w[0] - 1
            mdd[i] = (w / np.maximum.accumulate(w) - 1).min()
            low[i] = w.min() / w[0] - 1
        out[f"ret_{h}"] = ret
        out[f"maxdd_{h}"] = mdd
        out[f"lowest_{h}"] = low
    return pd.DataFrame(out, index=spx.index)


def warning_stats(sig: pd.DataFrame, fwd: pd.DataFrame, horizon: int = 126) -> pd.DataFrame:
    """Odds of a >=10% / >=20% decline within ``horizon`` days, by warning level, near the high only."""
    df = pd.concat([sig[["warning_level", "near_high"]], fwd[[f"maxdd_{horizon}", f"ret_{horizon}"]]], axis=1)
    df = df[df["near_high"]].dropna()
    rows = []
    for lvl in range(0, 6):
        sub = df[df["warning_level"] >= lvl] if lvl else df
        starts = episode_starts((sig["warning_level"] >= lvl) & sig["near_high"]) if lvl else []
        rows.append({
            "level": lvl,
            "days": len(sub),
            "share_of_time": len(sub) / len(df) if len(df) else np.nan,
            "episodes": len(starts) if lvl else np.nan,
            "p_dd10": (sub[f"maxdd_{horizon}"] <= -0.10).mean(),
            "p_dd20": (sub[f"maxdd_{horizon}"] <= -0.20).mean(),
            "avg_ret": sub[f"ret_{horizon}"].mean(),
        })
    return pd.DataFrame(rows)


def warning_episode_stats(sig: pd.DataFrame, fwd: pd.DataFrame, min_level: int, horizon: int = 126) -> dict:
    """Per-episode odds: of the times the warning first reached ``min_level`` near the high,
    how many were followed by a >=10% / >=20% decline within ``horizon`` days."""
    starts = episode_starts((sig["warning_level"] >= min_level) & sig["near_high"])
    dd = fwd[f"maxdd_{horizon}"].reindex(starts).dropna()
    return {"episodes": len(dd), "hit10": int((dd <= -0.10).sum()), "hit20": int((dd <= -0.20).sum())}


def correction_lead_table(sig: pd.DataFrame, levels=(2, 3), lookback_days: int = 60, min_depth: float = 0.10) -> pd.DataFrame:
    """For each decline of >= ``min_depth`` from a 1-year high: did a warning fire in the
    ``lookback_days`` before the market first crossed -10%, and how many days ahead?"""
    dd = sig["dd_1y"]
    in_corr = dd <= CORRECTION_DD
    crossings = episode_starts(in_corr, gap_days=TRADING_DAYS_YEAR // 2)
    rows = []
    for cross in crossings:
        i = sig.index.get_loc(cross)
        # depth of this decline: worst dd until it recovers above -5%
        after = dd.iloc[i:]
        rec = after[after > -0.05]
        end = rec.index[0] if len(rec) else after.index[-1]
        depth = dd.loc[cross:end].min()
        if depth > -min_depth:
            continue
        window = sig.iloc[max(0, i - lookback_days):i]
        row = {"crossed_-10%": cross.date(), "max_decline": depth}
        for lvl in levels:
            hits = window.index[window["warning_level"] >= lvl]
            row[f"lead_days_L{lvl}"] = (i - sig.index.get_loc(hits[0])) if len(hits) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def entry_stats(sig: pd.DataFrame, fwd: pd.DataFrame, spx: pd.Series) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Summary per entry signal vs baseline (all days >= 15% below the 1-year high), plus episode lists."""
    summary, episodes = [], {}
    baseline = sig.index[sig["dd_1y"] <= -0.15]
    groups = {"baseline": baseline}
    for name in ENTRY_SIGNALS:
        groups[name] = episode_starts(sig[f"e_{name}"])
    for name, dates in groups.items():
        f = fwd.reindex(dates)
        summary.append({
            "signal": name,
            "count": len(dates),
            "win_63": (f["ret_63"] > 0).mean(), "avg_63": f["ret_63"].mean(),
            "win_252": (f["ret_252"] > 0).mean(), "avg_252": f["ret_252"].mean(),
            "avg_further_drop_126": f["lowest_126"].mean(),
            "worst_further_drop_126": f["lowest_126"].min(),
        })
        if name != "baseline":
            rows = []
            for dt in dates:
                # true trough of this decline: from the prior 1-year high until price regains it
                i = spx.index.get_loc(dt)
                before = spx.iloc[max(0, i - TRADING_DAYS_YEAR):i + 1]
                peak_date, peak = before.idxmax(), before.max()
                after = spx.iloc[i:]
                regained = after[after >= peak]
                end = regained.index[0] if len(regained) else after.index[-1]
                trough = spx.loc[peak_date:end].idxmin()
                rows.append({
                    "date": dt.date(), "drawdown": sig.at[dt, "dd_1y"],
                    "days_vs_trough": int(i - spx.index.get_loc(trough)),
                    "further_drop_6m": fwd.at[dt, "lowest_126"], "ret_3m": fwd.at[dt, "ret_63"],
                    "ret_12m": fwd.at[dt, "ret_252"],
                })
            episodes[name] = pd.DataFrame(rows)
    return pd.DataFrame(summary), episodes


# ---------------------------------------------------------------------------
# Live state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalEvent:
    key: str
    kind: str  # "warning_up" | "warning_clear" | "entry"
    title: str
    message: str


def todays_events(sig: pd.DataFrame, fwd: pd.DataFrame, hold_days: int = HOLD_DAYS,
                  min_warning_level: int = 3) -> list[SignalEvent]:
    """Events for the last trading day, decided by comparing it with earlier days.

    Stateless: rerunning on the same data produces the same events. Warning bands
    are held for ``hold_days`` so one day's wobble does not send raise/clear pairs:
    a raise needs today's band above every band of the previous ``hold_days`` days,
    a clear needs ``hold_days`` consecutive normal days. Bands below
    ``min_warning_level`` are treated as normal (level 2 alone has had no edge).
    """
    if len(sig) < hold_days + 2:
        return []
    today = sig.iloc[-1]
    date = sig.index[-1].strftime("%Y-%m-%d")
    stats_sig, stats_fwd = sig.loc[STATS_START:], fwd.loc[STATS_START:]
    events: list[SignalEvent] = []

    floor = _band(min_warning_level)
    bands = sig["warning_level"].map(_band).where(sig["near_high"], 0)
    bands = bands.where(bands >= floor, 0)
    lvl, band = int(today["warning_level"]), int(bands.iloc[-1])
    prior_max = int(bands.iloc[-1 - hold_days:-1].max())
    if band > prior_max and band >= 1:
        active = [WARNING_CONDITIONS[c] for c in WARNING_CONDITIONS if today[f"w_{c}"]]
        st = warning_episode_stats(stats_sig, stats_fwd, min_level=lvl)
        base = stats_fwd["maxdd_126"][stats_sig["near_high"]].dropna()
        events.append(SignalEvent(
            f"warning-{date}-{lvl}", "warning_up",
            f"早期警示升到「{WARNING_LEVEL_NAMES[lvl]}」（{lvl}/5）",
            "觸發條件：" + "、".join(active) + "。\n"
            f"{STATS_START[:4]} 年以來，大盤接近高點時出現 ≥{lvl} 個條件共 {st['episodes']} 次，"
            f"之後 6 個月內跌 ≥10% 有 {st['hit10']} 次、跌 ≥20% 有 {st['hit20']} 次"
            f"（任一天的基準機率：跌 ≥10% {(base <= -0.10).mean():.0%}、跌 ≥20% {(base <= -0.20).mean():.0%}）。\n"
            f"目前距一年高點 {today['dd_1y']:+.1%}。這是風險升高的提醒，不是崩盤預測。",
        ))
    elif band == 0 and bands.iloc[-hold_days:].max() == 0 and bands.iloc[-1 - hold_days] >= 1:
        events.append(SignalEvent(f"warning-clear-{date}", "warning_clear", "早期警示解除",
                                  f"連續 {hold_days} 個交易日回到正常（目前 {lvl}/5）。"))

    for name, label in ENTRY_SIGNALS.items():
        col = f"e_{name}"
        # Fire on the first day of a new episode only.
        recent = sig[col].iloc[-61:-1]
        if today[col] and not recent.any():
            starts = episode_starts(stats_sig[col])[:-1]  # exclude today
            f = stats_fwd.reindex(starts).dropna(subset=["ret_252"])
            if len(f):
                history = (f"{STATS_START[:4]} 年以來 {len(f)} 次：12 個月後上漲 {(f['ret_252'] > 0).sum()} 次、"
                           f"平均 {f['ret_252'].mean():+.1%}；訊號後 6 個月內平均再跌 {f['lowest_126'].mean():.1%}、"
                           f"最多再跌 {f['lowest_126'].min():.1%}。")
            else:
                history = f"{STATS_START[:4]} 年以來這個指數還沒有可比較的先例，沒有歷史勝率可參考。"
            events.append(SignalEvent(
                f"entry-{name}-{date}", "entry", f"進場訊號：{label.split('：')[0]}",
                f"{label}。目前距一年高點 {today['dd_1y']:+.1%}，VIX {today['vix']:.1f}。\n{history}\n"
                "訊號不代表就是最低點，歷史上常在訊號後再探底，分批進場較能承受。",
            ))
    return events


def position_frame(spx: pd.Series, sig: pd.DataFrame, warning_level: int = 4) -> pd.DataFrame:
    """Rule exposure of the backtested "combo B" strategy (scripts/run_overlay_study.py).

    trend      Faber 10-month average rule (1 = above, 0 = below), decided at the close of a month's
               last trading day; mid-month days keep the previous decision. Every decision is traded
               at the next close, as in the backtest.
    warning    0.5 while the early warning (>= ``warning_level``) is active, else 1.
    target     trend x warning, but at least 0.5 after an entry signal while the trend is out.
    ma10_gap   price versus the provisional 10-month average (9 previous month-ends + today).
    attrs["exit_stats"]  what happened after past exits on this index (see ``exit_stats``).
    """
    from .overlay import combined_exposure, faber_exposure, warning_overlay

    p = spx.dropna()
    trend = faber_exposure(p)
    # faber_exposure treats the last row as a month-end decision. That is only final if today is the
    # month's last trading day; otherwise today keeps the previous decision (no mid-month flips).
    if len(trend) > 1 and not is_last_trading_day_of_month(p.index[-1]):
        trend.iloc[-1] = trend.iloc[-2]
    warn = warning_overlay(sig, warning_level)
    entry = sig[[f"e_{k}" for k in ENTRY_SIGNALS]].fillna(False).astype(bool).any(axis=1)
    target = combined_exposure(trend, warn, entry)

    monthly = p.resample("ME").last()
    ma10 = monthly.rolling(10, min_periods=10).mean()
    # Provisional 10-month average on each day: the 9 previous month-end closes plus today's close.
    prev9_sum = monthly.rolling(9, min_periods=9).sum().shift(1)
    month_of_day = p.index.to_period("M")
    prev9 = pd.Series(prev9_sum.values, index=prev9_sum.index.to_period("M")).reindex(month_of_day).values
    gap = p / ((prev9 + p.values) / 10) - 1
    out = pd.DataFrame({"trend": trend, "warning": warn, "target": target, "ma10_gap": gap}, index=p.index)
    out.attrs["ma10_provisional"] = float(ma10.iloc[-1])
    out.attrs["price"] = float(p.iloc[-1])
    out.attrs["exit_stats"] = exit_stats(p)
    return out


def is_last_trading_day_of_month(day: pd.Timestamp) -> bool:
    """True when the next weekday falls in a new month. Exchange holidays on the month's last
    weekday (rare) make this miss by one day; the flip then shows on the first day of the new month."""
    return (day + pd.offsets.BDay(1)).month != day.month


def exit_stats(price: pd.Series) -> dict:
    """History of the 10-month-average exit on this index: what happened after each exit.

    saved: the rule bought back cheaper than it sold (the exit avoided a decline).
    further_drop: lowest close within a year after the exit, versus the exit price.
    """
    from .overlay import faber_exposure

    p = price.dropna()
    trend = faber_exposure(p)
    change = trend.diff().fillna(0)
    exits, entries = change.index[change < 0], change.index[change > 0]
    rows = []
    for d in exits:
        later = entries[entries > d]
        p0 = p.loc[d]
        back = p.loc[later[0]] / p0 - 1 if len(later) else np.nan
        rows.append({"date": d, "reentry_vs_exit": back,
                     "further_drop": p.loc[d:].iloc[:253].min() / p0 - 1})
    df = pd.DataFrame(rows)
    done = df.dropna(subset=["reentry_vs_exit"]) if len(df) else df
    return {
        "exits": len(df), "since": p.index[0].year,
        "saved": int((done["reentry_vs_exit"] < 0).sum()) if len(done) else 0, "closed": len(done),
        "median_reentry_vs_exit": float(done["reentry_vs_exit"].median()) if len(done) else np.nan,
        "worst_whipsaw": float(done["reentry_vs_exit"].max()) if len(done) else np.nan,
        "median_further_drop": float(df["further_drop"].median()) if len(df) else np.nan,
        "big_drops_avoided": int((df["further_drop"] <= -0.20).sum()) if len(df) else 0,
    }


def position_events(pos: pd.DataFrame, date_label: str,
                    evidence: str = "S&P 500 1990–2026：年化報酬與一直持有相近、最大回撤 -21% 對 -55%") -> list[SignalEvent]:
    if len(pos) < 2:
        return []
    today, prev = pos.iloc[-1], pos.iloc[-2]
    events = []
    st = pos.attrs.get("exit_stats") or {}
    if today["trend"] < prev["trend"]:
        history = ""
        if st.get("exits"):
            history = (f"{st['since']} 年以來這個指數出現過 {st['exits']} 次出場訊號：之後一年內跌幅中位數 "
                       f"{st['median_further_drop']:+.1%}，其中 {st['big_drops_avoided']} 次之後又跌超過 20%；"
                       f"已重新進場的 {st['closed']} 次中，{st['saved']} 次買回價比賣出價低（真的避開下跌），"
                       f"最差一次買回價比賣出價高 {st['worst_whipsaw']:+.1%}（被洗出場）。\n")
        events.append(SignalEvent(
            f"exit-{date_label}", "exit", "出場訊號：月底收盤跌破 10 個月均線",
            "本月最後一個交易日收盤跌破 10 個月均線，照規則在下一個交易日收盤減碼。\n" + history
            + "出場訊號一定晚於最高點：觸發時指數通常已從高點跌了一段（SOX 歷史上是 13–39%）。它的目的是避開之後更大的跌幅，"
            "不是賣在最高點。這是規則的機械輸出，不是投資建議。",
        ))
    elif today["trend"] > prev["trend"]:
        events.append(SignalEvent(
            f"trend-{date_label}", "trend", "重新進場訊號：月底收盤站回 10 個月均線",
            "本月最後一個交易日收盤回到 10 個月均線之上，照規則在下一個交易日收盤恢復持有。",
        ))
    elif today["trend"] >= 1 and today["ma10_gap"] < 0 and not (pos["ma10_gap"].iloc[-11:-1] < 0).any():
        events.append(SignalEvent(
            f"exit-watch-{date_label}", "exit_watch", "出場預警：價格跌到 10 個月均線之下",
            f"目前價格比 10 個月均線低 {abs(today['ma10_gap']):.1%}。這還不是出場訊號："
            "要等這個月最後一個交易日收盤仍在均線之下，當天收盤後才會正式發出出場訊號。",
        ))
    if today["target"] != prev["target"]:
        events.append(SignalEvent(
            f"position-{date_label}-{today['target']:.2f}", "position",
            f"規則部位調整：{prev['target']:.0%} → {today['target']:.0%}",
            f"依回測過的「均線 × 警示 + 恐慌進場」規則（{evidence}）。照規則在下一個交易日收盤調整。"
            "這是規則的機械輸出，不是投資建議。",
        ))
    return events


def _band(level: int) -> int:
    return 0 if level <= 1 else (1 if level == 2 else 2)
