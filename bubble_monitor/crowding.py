"""Investor crowding measures and their relation to later S&P 500 returns.

Hypothesis under test: when (retail) investors are heavily positioned in stocks,
later returns are weaker and large drawdowns more likely.

Measures (fixed before testing):
    household_alloc  Fed Z.1: households' directly + indirectly held equities as a
                     % of their financial assets (retail positioning), 1945+
    aiae             Aggregate investor allocation to equities: corporate equity
                     market value / (equities + total debt), Philosophical Economics 2013
    margin_yoy       FINRA customer margin debit balances, 12-month growth (retail leverage), 1997+
    margin_to_cash   FINRA margin debt / free credit balances (borrowed vs idle cash), 1997+
    mmf_to_equity    money market fund assets / equity market value; low = little cash
                     on the sidelines. Sign is flipped so that higher = more crowded.

Each measure is aligned to when it was published (Z.1 ~10 weeks after quarter end,
FINRA ~3 weeks after month end), and crowding is ranked against its own past only
(expanding percentile), so the quintile results are point-in-time.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

Z1_LAG = pd.DateOffset(months=3, days=70)
FINRA_LAG = pd.DateOffset(months=1, days=25)

MEASURES = {
    "household_alloc": ("家庭股票配置比例", "Fed Z.1 家庭持股占金融資產 %（散戶部位）"),
    "aiae": ("投資人股票配置（AIAE）", "股票市值 ÷（股票 + 全體債務）"),
    "margin_yoy": ("融資餘額年增率", "FINRA 融資餘額 12 個月成長率（散戶槓桿）"),
    "margin_to_cash": ("融資 ÷ 閒置現金", "FINRA 融資餘額 ÷ 帳戶閒置現金"),
    "mmf_to_equity": ("場邊現金（反向）", "貨幣基金 ÷ 股市市值，取負號：越高 = 場邊現金越少"),
}


def _published(series: pd.Series, period_start_lag: pd.DateOffset) -> pd.Series:
    s = series.dropna()
    return pd.Series(s.values, index=s.index + period_start_lag).sort_index()


def asof_monthly(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    s = series.dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(s.index.union(index)).ffill().reindex(index)


def build_measures(raw: Mapping[str, pd.Series], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Month-end values of each crowding measure, using only data published by then."""
    m = pd.DataFrame(index=index)
    m["household_alloc"] = asof_monthly(_published(raw["household_alloc"], Z1_LAG), index)

    eq, debt = raw["corp_equity"].dropna(), raw["total_debt"].dropna()
    aiae = (eq / (eq + debt.reindex(eq.index))).dropna() * 100
    m["aiae"] = asof_monthly(_published(aiae, Z1_LAG), index)

    mmf = raw["mmf"].dropna()
    m["mmf_to_equity"] = -asof_monthly(_published((mmf / eq.reindex(mmf.index)).dropna() * 100, Z1_LAG), index)

    debit = raw.get("margin_debt")
    if debit is not None and len(debit.dropna()):
        debit = debit.dropna()
        m["margin_yoy"] = asof_monthly(_published(debit / debit.shift(12) - 1, FINRA_LAG), index) * 100
        cash = raw["free_credit"].reindex(debit.index)
        m["margin_to_cash"] = asof_monthly(_published((debit / cash).dropna(), FINRA_LAG), index)
    else:
        m["margin_yoy"] = np.nan
        m["margin_to_cash"] = np.nan
    return m


def expanding_percentile(s: pd.Series, min_periods: int = 60) -> pd.Series:
    """Rank of each value among all values up to and including it (no look-ahead)."""
    return s.expanding(min_periods=min_periods).apply(
        lambda a: (a[~np.isnan(a)] <= a[-1]).mean() if not np.isnan(a[-1]) else np.nan, raw=True
    ).where(s.notna())


def forward_monthly(spx_m: pd.Series, horizons=(3, 12, 36)) -> pd.DataFrame:
    """Forward price return and forward max drawdown on month-end closes."""
    out = {}
    p = spx_m.values.astype(float)
    n = len(p)
    for h in horizons:
        ret = np.full(n, np.nan)
        mdd = np.full(n, np.nan)
        for i in range(n - h):
            w = p[i:i + h + 1]
            ret[i] = w[-1] / w[0] - 1
            mdd[i] = (w / np.maximum.accumulate(w) - 1).min()
        out[f"ret_{h}"] = ret
        out[f"maxdd_{h}"] = mdd
    return pd.DataFrame(out, index=spx_m.index)


def rank_corr(x: pd.Series, y: pd.Series, step: int = 1) -> tuple[float, int]:
    """Spearman correlation on every ``step``-th month (step = horizon gives non-overlapping samples)."""
    df = pd.concat([x, y], axis=1).dropna().iloc[::step]
    if len(df) < 8:
        return np.nan, len(df)
    return float(df.iloc[:, 0].rank().corr(df.iloc[:, 1].rank())), len(df)


def quintile_table(pct: pd.Series, fwd: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    df = pd.concat([pct.rename("pct"), fwd[[f"ret_{horizon}", f"maxdd_{horizon}"]]], axis=1).dropna()
    df["q"] = pd.cut(df["pct"], [0, 0.2, 0.4, 0.6, 0.8, 1.0001], labels=["Q1 最不擁擠", "Q2", "Q3", "Q4", "Q5 最擁擠"],
                     include_lowest=True)
    g = df.groupby("q", observed=False)
    return pd.DataFrame({
        "months": g.size(),
        "avg_ret": g[f"ret_{horizon}"].mean(),
        "p_up": g[f"ret_{horizon}"].apply(lambda r: (r > 0).mean() if len(r) else np.nan),
        "avg_maxdd": g[f"maxdd_{horizon}"].mean(),
        "p_dd20": g[f"maxdd_{horizon}"].apply(lambda r: (r <= -0.20).mean() if len(r) else np.nan),
    })
