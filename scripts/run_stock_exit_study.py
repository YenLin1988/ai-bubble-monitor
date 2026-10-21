"""Which exit rule is credible for volatile individual stocks? Pre-registered test.

Universe: 45 high-volatility US stocks, deliberately mixing big winners with names that crashed and
never fully recovered (CSCO/INTC after 2000, PTON, ZM, UPST, PLUG...), plus the user's holdings.
Free data has no delisted stocks, so survivorship bias remains (results lean optimistic for buy-and-hold).

Rules (fixed before testing):
    S1 10-month average (month-end)         S4 chandelier: high since entry - 3 x 22-day avg range,
    S2 200-day average (daily)                 re-enter on a new 50-day high
    S3 50-day average (daily)               S5 20% trailing stop, re-enter on a new 50-day high

Credibility criteria (all must hold):
    1. median max-drawdown improvement vs buy-and-hold >= 15 percentage points
    2. Sharpe above buy-and-hold in >= 60% of stocks
    3. Sharpe above buy-and-hold in a majority of stocks in BOTH 2000-2012 and 2013-2026

Usage:
    python scripts/run_stock_exit_study.py [--refresh]
Writes reports/stock_exit_study.html.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bubble_monitor import overlay as ov  # noqa: E402
from bubble_monitor.data_sources import load_many  # noqa: E402

CACHE_DIR = ROOT / "data" / "backtest_cache"
REPORT_DIR = ROOT / "reports"
HOLDINGS = ["NBIS", "HOOD", "SPCX", "AAOI", "AXTI", "VST"]
UNIVERSE = HOLDINGS + [
    # dot-com era winners that crashed
    "CSCO", "INTC", "QCOM", "ORCL", "AMZN", "EBAY", "AMD", "MU", "NVDA", "AMAT", "NOK", "VIAV",
    # 2010s-2020s high flyers, both survivors and busts
    "TSLA", "NFLX", "SHOP", "ROKU", "ZM", "PTON", "COIN", "PLTR", "SNOW", "AFRM", "UPST", "MRNA",
    "ENPH", "SEDG", "PLUG", "FSLR", "RIVN", "SMCI", "CRWD", "NET", "DDOG", "MSTR", "LITE", "COHR",
    "TTD", "CVNA", "SOFI",
]
RULES = {"S1": "10 個月均線", "S2": "200 日均線", "S3": "50 日均線", "S4": "吊燈停損 3×波幅", "S5": "20% 移動停損"}
HALVES = {"2000–2012": ("2000-01-01", "2012-12-31"), "2013–2026": ("2013-01-01", "2026-12-31")}
COST = 0.002  # per unit of exposure traded; individual stocks have wider spreads than indexes


def _pct(x, d=1):
    return "" if pd.isna(x) else f"{x * 100:.{d}f}%"


def tbl(df: pd.DataFrame) -> str:
    return df.to_html(index=False, escape=False, border=0, classes="tbl")


def exposures(p: pd.Series) -> dict[str, pd.Series]:
    return {"S1": ov.faber_exposure(p), "S2": ov.sma_exposure(p, 200), "S3": ov.sma_exposure(p, 50),
            "S4": ov.chandelier_exposure(p, 3.0, 22, 50), "S5": ov.pct_trailing_exposure(p, 0.20, 50)}


def exit_quality(p: pd.Series, exp: pd.Series) -> dict:
    """How much was given back from the in-trade peak at each exit, and how often the exit was a whipsaw
    (bought back higher than it sold)."""
    change = exp.diff().fillna(0)
    exits, entries = change.index[change < 0], change.index[change > 0]
    given_back, whips, n_closed = [], 0, 0
    last_entry = p.index[0]
    for d in exits:
        prev_entries = entries[entries < d]
        start = prev_entries[-1] if len(prev_entries) else last_entry
        peak = p.loc[start:d].max()
        given_back.append(p.loc[d] / peak - 1)
        later = entries[entries > d]
        if len(later):
            n_closed += 1
            whips += p.loc[later[0]] > p.loc[d]
    return {"exits": len(exits), "median_given_back": float(np.median(given_back)) if given_back else np.nan,
            "whipsaw_rate": whips / n_closed if n_closed else np.nan}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 230)

    raw = load_many({"tbill": "DTB3"}, {t: t for t in UNIVERSE}, CACHE_DIR, args.refresh)
    cash = raw["tbill"]
    rows = []
    for t in UNIVERSE:
        p = raw.get(t)
        if p is None or len(p.dropna()) < 300:
            print("skip", t, "too short")
            continue
        p = p.dropna()
        p = p[p > 0].loc["1995-01-01":]
        start = p.index[0] + pd.DateOffset(days=300)  # warm-up for the 200-day / 10-month rules
        sims = {"BH": ov.simulate(p.loc[start:], pd.Series(1.0, index=p.index), cash, COST)}
        exps = exposures(p)
        for k, e in exps.items():
            sims[k] = ov.simulate(p.loc[start:], e, cash, COST)
        for period, (a, b) in {"全期": (str(start.date()), "2026-12-31"), **HALVES}.items():
            for k, sim in sims.items():
                sub = sim.loc[max(pd.Timestamp(a), start):b].copy()
                if len(sub) < 250:
                    continue
                sub["equity"] = (1 + sub["ret"]).cumprod()
                m = ov.metrics(sub, cash)
                row = {"ticker": t, "period": period, "rule": k, "years": len(sub) / 252, **m}
                if period == "全期" and k != "BH":
                    row.update(exit_quality(p.loc[start:], exps[k].loc[start:]))
                rows.append(row)
    res = pd.DataFrame(rows)

    # ---- per-rule summary and verdict ----
    def paired(period, k):
        a = res[(res["period"] == period) & (res["rule"] == k)].set_index("ticker")
        b = res[(res["period"] == period) & (res["rule"] == "BH")].set_index("ticker")
        j = a.join(b, rsuffix="_bh", how="inner")
        return j

    summary, verdict = [], []
    for k, label in RULES.items():
        j = paired("全期", k)
        dd_gain = (j["max_dd"] - j["max_dd_bh"]).median()
        share = (j["sharpe"] > j["sharpe_bh"]).mean()
        halves = {h: (paired(h, k)["sharpe"] > paired(h, k)["sharpe_bh"]).mean() for h in HALVES}
        halves_n = {h: len(paired(h, k)) for h in HALVES}
        c1, c2, c3 = dd_gain >= 0.15, share >= 0.60, all(v > 0.5 for v in halves.values())
        summary.append({"規則": f"{k} {label}", "個股數": len(j),
                        "年化報酬中位數": _pct(j["cagr"].median()), "持有中位數": _pct(j["cagr_bh"].median()),
                        "最大回撤中位數": _pct(j["max_dd"].median()), "持有回撤中位數": _pct(j["max_dd_bh"].median()),
                        "夏普勝過持有": _pct(share, 0),
                        **{f"{h} 夏普勝過持有": f"{_pct(v, 0)}（{halves_n[h]} 檔）" for h, v in halves.items()},
                        "出場時已從高點回落（中位數）": _pct(j["median_given_back"].median()),
                        "被洗出場比例": _pct(j["whipsaw_rate"].median(), 0),
                        "每年出場次數": f"{(j['exits'] / j['years']).median():.1f}"})
        verdict.append({"規則": f"{k} {label}", "1. 回撤少 ≥15 個百分點": f"{'✅' if c1 else '❌'} {_pct(dd_gain)}",
                        "2. ≥60% 個股夏普勝過持有": f"{'✅' if c2 else '❌'} {_pct(share, 0)}",
                        "3. 兩段期間都過半": f"{'✅' if c3 else '❌'} " + " / ".join(_pct(v, 0) for v in halves.values()),
                        "可信": "✅" if (c1 and c2 and c3) else "❌", "_sharpe": j["sharpe"].median(),
                        "_ok": c1 and c2 and c3})
    summary, verdict = pd.DataFrame(summary), pd.DataFrame(verdict)
    print(summary.to_string(index=False), "\n")
    print(verdict.drop(columns=["_sharpe", "_ok"]).to_string(index=False), "\n")
    ok = verdict[verdict["_ok"]]
    best = ok.sort_values("_sharpe", ascending=False)["規則"].iloc[0] if len(ok) else None
    print("best credible rule:", best)

    # ---- holdings detail ----
    hold = res[(res["period"] == "全期") & res["ticker"].isin(HOLDINGS)].copy()
    print(hold[["ticker", "rule", "cagr", "max_dd", "sharpe", "time_defensive", "median_given_back", "whipsaw_rate"]]
          .round(3).to_string(index=False))
    hold_html = tbl(pd.DataFrame({
        "股票": hold["ticker"], "規則": hold["rule"].map({"BH": "一直持有", **{k: f"{k} {v}" for k, v in RULES.items()}}),
        "年化報酬": hold["cagr"].map(_pct), "最大回撤": hold["max_dd"].map(_pct), "夏普": hold["sharpe"].map(lambda x: f"{x:.2f}"),
        "空手時間": hold["time_defensive"].map(lambda x: _pct(x, 0)),
        "出場時已從高點回落": hold["median_given_back"].map(_pct)}))

    html = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>個股出場規則</title>
<style>
body{{background:#0D1117;color:#E6EDF3;font-family:-apple-system,"Segoe UI","Noto Sans TC",sans-serif;max-width:1150px;margin:0 auto;padding:16px;line-height:1.6}}
h1{{font-size:24px}} h2{{font-size:19px;margin-top:36px;border-bottom:1px solid #30363D;padding-bottom:6px}}
.tbl{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums;display:block;overflow-x:auto}}
.tbl th,.tbl td{{padding:6px 10px;border-bottom:1px solid #21262D;text-align:right;white-space:nowrap}}
.tbl th:first-child,.tbl td:first-child{{text-align:left}} .tbl th{{color:#8B949E;font-weight:600}}
.note{{color:#8B949E;font-size:13px}}
</style></head><body>
<h1>高波動個股：哪個出場規則可信？</h1>
<p class="note">{len(set(res['ticker']))} 檔高波動美股（含大漲與崩跌未回的個股、以及你的持股），規則與標準測試前固定。
訊號收盤確認、隔天收盤交易，每次進出扣 0.2% 成本，空手賺國庫券利率。免費資料沒有已下市股票，仍有存活者偏誤（對「一直持有」偏有利）。</p>
<h2>結論</h2>{tbl(verdict.drop(columns=['_sharpe', '_ok']))}
<p class="note">三項都通過且夏普中位數最高的規則：<b>{best or '沒有規則通過'}</b></p>
<h2>各規則摘要（個股中位數）</h2>{tbl(summary)}
<h2>你的持股</h2>{hold_html}
<p class="note">歷史回測不保證未來，這不是投資建議。</p></body></html>"""
    REPORT_DIR.mkdir(exist_ok=True)
    out = REPORT_DIR / "stock_exit_study.html"
    out.write_text(html, encoding="utf-8")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
