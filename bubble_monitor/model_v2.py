"""v2 stress model: regime-independent factor definitions.

Why v2 exists (see reports/backtest_report.html): the v1 model scores absolute
levels against fixed thresholds. Over 1986-2026 that made several factors sit
at 0 or 100 for decades (yield curve, JPY, net liquidity, consumer sentiment),
let the composite drift with the interest-rate era, and rewarded lagging data
(unemployment, M2) that only rises after a crash.

v2 rules, chosen from economic reasoning rather than fitted to the backtest:
1. Levels become changes or trailing-10Y percentiles, so a factor reads
   "unusual versus the recent decade" instead of "above a 2024-era number".
   Percentiles only count once they are extreme: median -> 0, 95th pct -> 100.
2. The yield curve scores the inversion -> re-steepening sequence, the
   classic pre-recession pattern, instead of punishing a normal positive curve.
3. Lagging macro (M2, unemployment, consumer sentiment) is kept at half weight;
   unemployment uses the Sahm-rule rise rather than its level.
4. New faster market-internal signals: credit-spread velocity, small-cap
   breadth, and VIX term-structure inversion.

All functions take the monthly value frame from ``backtest.build_factor_values``
(full history, oldest first) and only look backwards, so they are safe to
evaluate month by month.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

PCT_WINDOW_MONTHS = 120
PCT_MIN_MONTHS = 60


def trailing_percentile(s: pd.Series, window: int | None = PCT_WINDOW_MONTHS, min_periods: int = PCT_MIN_MONTHS) -> pd.Series:
    """Percentile (0-1) of each value within its trailing window; ``window=None`` = expanding."""
    s = s.astype(float)
    roll = s.expanding(min_periods=min_periods) if window is None else s.rolling(window, min_periods=min_periods)
    out = roll.apply(lambda a: (a[~np.isnan(a)] <= a[-1]).mean() if not np.isnan(a[-1]) else np.nan, raw=True)
    return out.where(s.notna())


def extreme_score(pct: pd.Series, start: float = 0.50, full: float = 0.95) -> pd.Series:
    """Map a percentile onto 0-100 so only unusually high readings add stress."""
    return ((pct - start) / (full - start) * 100).clip(0, 100)


def linear_score(s: pd.Series, safe: float, danger: float) -> pd.Series:
    return ((s - safe) / (danger - safe) * 100).clip(0, 100)


def _pct_score(s: pd.Series, **kw) -> pd.Series:
    return extreme_score(trailing_percentile(s, **kw))


def curve_uninversion(curve: pd.Series, memory_months: int = 24) -> pd.Series:
    """0 = no recent inversion, 50 = currently inverted, 100 = re-steepened after an inversion.

    Recessions and bear markets have historically started after the curve
    un-inverts, not while it is inverted, so the re-steepening phase scores highest.
    """
    recent_min = curve.rolling(memory_months, min_periods=1).min()
    score = pd.Series(np.where(recent_min >= 0, 0.0, np.where(curve < 0, 50.0, 100.0)), index=curve.index)
    return score.where(curve.notna())


def sahm_gap(unrate: pd.Series) -> pd.Series:
    """Sahm rule: 3m average unemployment minus its low of the prior 12 months."""
    avg3 = unrate.rolling(3, min_periods=3).mean()
    return avg3 - avg3.shift(1).rolling(12, min_periods=12).min()


def pct_change(s: pd.Series, months: int) -> pd.Series:
    return s / s.shift(months) - 1


Rule = Callable[[pd.DataFrame], pd.Series]

# name -> (zh label, cluster, base weight, description, rule)
V2_FACTORS: dict[str, tuple[str, str, float, str, Rule]] = {
    "Rate Shock": ("利率衝擊", "A 貨幣流動性", 1.0, "10Y 殖利率 12 個月變動的 10 年百分位",
                   lambda v: _pct_score(v["tnx"].diff(12))),
    "Fed Tightening": ("升息循環", "A 貨幣流動性", 1.0, "聯邦基金利率 12 個月變動的 10 年百分位",
                       lambda v: _pct_score(v["fed_funds"].diff(12))),
    "Liquidity Drain": ("流動性抽離", "A 貨幣流動性", 1.0, "淨流動性 6 個月跌幅的 10 年百分位",
                        lambda v: _pct_score(-pct_change(v["net_liquidity"], 6))),
    "JPY Carry Unwind": ("日圓套利平倉", "A 貨幣流動性", 1.0, "日圓 3 個月升值幅度的 10 年百分位",
                         lambda v: _pct_score(-pct_change(v["usdjpy"], 3))),
    "M2 Slowdown": ("貨幣供給放緩", "A 貨幣流動性", 0.5, "M2 年增率越低越高分（10 年百分位，落後指標半權重）",
                    lambda v: _pct_score(-v["m2_yoy"])),
    "Curve Un-inversion": ("殖利率曲線解除倒掛", "B 信用條件", 1.0, "24 個月內曾倒掛：倒掛中 50、重新轉正 100",
                           lambda v: curve_uninversion(v["curve"])),
    "Credit Spread": ("信用利差水準", "B 信用條件", 1.0, "高收益債利差的 10 年百分位",
                      lambda v: _pct_score(v["spread"])),
    "Credit Velocity": ("信用利差速度", "B 信用條件", 1.0, "高收益債利差 3 個月變動的 10 年百分位",
                        lambda v: _pct_score(v["spread"].diff(3))),
    "NFCI": ("金融環境", "B 信用條件", 1.0, "芝加哥 Fed 金融狀況指數（本身已標準化，-0.5→0、+0.5→100）",
             lambda v: linear_score(v["nfci"], -0.5, 0.5)),
    "Buffett Indicator": ("巴菲特指標", "C 估值情緒", 1.0, "股市市值 / GDP 的歷史（擴張視窗）百分位",
                          lambda v: _pct_score(v["buffett"], window=None, min_periods=120)),
    "Volatility": ("波動率", "C 估值情緒", 1.0, "VXN（2001 前用 VIX），15→0、35→100",
                   lambda v: linear_score(v["vxn"], 15, 35)),
    "Sentiment Drop": ("消費者信心下滑", "C 估值情緒", 0.5, "密西根信心 12 個月跌幅的 10 年百分位（半權重）",
                       lambda v: _pct_score(-v["umcsent"].diff(12))),
    "Dollar Surge": ("美元急升", "D 總體通膨", 1.0, "美元指數 6 個月漲幅的 10 年百分位",
                     lambda v: _pct_score(v["dxy_chg6"])),
    "Oil Shock": ("油價衝擊", "D 總體通膨", 1.0, "WTI 6 個月漲幅的 10 年百分位",
                  lambda v: _pct_score(pct_change(v["oil_m"], 6))),
    "Sahm Rule": ("薩姆規則", "D 總體通膨", 0.5, "失業率 3 月均值較前 12 個月低點上升 0→0.5pp（半權重）",
                  lambda v: linear_score(sahm_gap(v["unrate"]), 0.0, 0.5)),
    "Breadth": ("市場廣度", "E 市場內部", 1.0, "小型股相對 S&P 500 6 個月落後幅度的 10 年百分位",
                lambda v: _pct_score(-pct_change(v["rut_rel"], 6))),
    "VIX Term Structure": ("VIX 期限結構", "E 市場內部", 1.0, "VIX / VIX3M，0.90→0、1.05（倒掛）→100",
                           lambda v: linear_score(v["vix_ts"], 0.90, 1.05)),
}

V2_WEIGHTS = {name: spec[2] / sum(s[2] for s in V2_FACTORS.values()) for name, spec in V2_FACTORS.items()}
V2_LABELS_ZH = {name: spec[0] for name, spec in V2_FACTORS.items()}
V2_CLUSTERS = {name: spec[1] for name, spec in V2_FACTORS.items()}


def score_factors_v2(values: pd.DataFrame) -> pd.DataFrame:
    """Score every v2 factor; a factor whose inputs are missing yields NaN."""
    out = {}
    for name, (*_, rule) in V2_FACTORS.items():
        try:
            out[name] = rule(values).astype(float)
        except KeyError:
            out[name] = pd.Series(np.nan, index=values.index)
    return pd.DataFrame(out, index=values.index)
