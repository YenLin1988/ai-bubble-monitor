"""Second hypothesis: time volatile individual stocks with a MARKET-level exit signal.

The first study (run_stock_exit_study.py) found no stock-level exit rule credible. This tests one more
pre-declared idea and then stops: hold each stock only while an index trend rule says "in".
    M1 SOX 10-month average            M3 SOX "combo B" (trend x warning>=4 + capitulation re-entry),
    M2 Nasdaq-100 10-month average        i.e. the rule position in the daily email
Same universe, costs and credibility criteria as the first study.

Usage:
    python scripts/run_market_exit_study.py [--refresh]
Writes reports/market_exit_study.html.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bubble_monitor import daily_signals as ds  # noqa: E402
from bubble_monitor import overlay as ov  # noqa: E402
from bubble_monitor.data_sources import load_many  # noqa: E402
from run_stock_exit_study import COST, HALVES, HOLDINGS, UNIVERSE, exit_quality  # noqa: E402

CACHE_DIR = ROOT / "data" / "backtest_cache"
REPORT_DIR = ROOT / "reports"
RULES = {"M1": "SOX 10 個月均線", "M2": "那斯達克 100 10 個月均線", "M3": "SOX 均線 × 警示 + 恐慌進場（Email 規則部位）"}


def _pct(x, d=1):
    return "" if pd.isna(x) else f"{x * 100:.{d}f}%"


def tbl(df: pd.DataFrame) -> str:
    return df.to_html(index=False, escape=False, border=0, classes="tbl")


def on_dates(exp: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Index exposure decided at the index close, carried onto the stock's trading days."""
    return exp.reindex(exp.index.union(dates)).ffill().reindex(dates).fillna(1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 230)

    raw = load_many({"tbill": "DTB3", "hy": "BAMLH0A0HYM2", "baa10y": "BAA10Y", "t10y2y": "T10Y2Y"},
                    {"spx": "^GSPC", "sox": "^SOX", "ndx": "^NDX", "rut": "^RUT", "vix_daily": "^VIX",
                     "vix3m_daily": "^VIX3M", **{t: t for t in UNIVERSE}}, CACHE_DIR, args.refresh)
    cash = raw["tbill"]
    d = ds.build_daily_frame(raw)
    sox, ndx = raw["sox"].dropna(), raw["ndx"].dropna()
    sig = ds.compute_signals(d, sox)
    entry = sig[["e_capitulation", "e_combo"]].fillna(False).astype(bool).any(axis=1)
    sox_trend = ov.faber_exposure(sox)
    market = {
        "M1": sox_trend,
        "M2": ov.faber_exposure(ndx),
        "M3": ov.combined_exposure(sox_trend, ov.warning_overlay(sig.dropna(subset=["dd_1y"]), 4), entry),
    }

    rows = []
    for t in UNIVERSE:
        p = raw.get(t)
        if p is None or len(p.dropna()) < 300:
            print("skip", t)
            continue
        p = p.dropna()
        p = p[p > 0].loc["1995-01-01":]
        start = p.index[0] + pd.DateOffset(days=300)  # same window as the first study
        sims = {"BH": ov.simulate(p.loc[start:], pd.Series(1.0, index=p.index), cash, COST)}
        exps = {k: on_dates(e, p.index) for k, e in market.items()}
        for k, e in exps.items():
            sims[k] = ov.simulate(p.loc[start:], e, cash, COST)
        for period, (a, b) in {"全期": (str(start.date()), "2026-12-31"), **HALVES}.items():
            for k, sim in sims.items():
                sub = sim.loc[max(pd.Timestamp(a), start):b].copy()
                if len(sub) < 250:
                    continue
                sub["equity"] = (1 + sub["ret"]).cumprod()
                row = {"ticker": t, "period": period, "rule": k, "years": len(sub) / 252, **ov.metrics(sub, cash)}
                if period == "全期" and k != "BH":
                    row.update(exit_quality(p.loc[start:], exps[k].loc[start:]))
                rows.append(row)
    res = pd.DataFrame(rows)

    def paired(period, k):
        a = res[(res["period"] == period) & (res["rule"] == k)].set_index("ticker")
        b = res[(res["period"] == period) & (res["rule"] == "BH")].set_index("ticker")
        return a.join(b, rsuffix="_bh", how="inner")

    summary, verdict = [], []
    for k, label in RULES.items():
        j = paired("全期", k)
        dd_gain = (j["max_dd"] - j["max_dd_bh"]).median()
        share = (j["sharpe"] > j["sharpe_bh"]).mean()
        halves = {h: (paired(h, k)["sharpe"] > paired(h, k)["sharpe_bh"]).mean() for h in HALVES}
        c1, c2, c3 = dd_gain >= 0.15, share >= 0.60, all(v > 0.5 for v in halves.values())
        summary.append({"規則": f"{k} {label}", "個股數": len(j),
                        "年化報酬中位數": _pct(j["cagr"].median()), "持有中位數": _pct(j["cagr_bh"].median()),
                        "最大回撤中位數": _pct(j["max_dd"].median()), "持有回撤中位數": _pct(j["max_dd_bh"].median()),
                        "夏普中位數": f"{j['sharpe'].median():.2f}", "持有夏普中位數": f"{j['sharpe_bh'].median():.2f}",
                        "出場時個股已從高點回落": _pct(j["median_given_back"].median()),
                        "被洗出場比例": _pct(j["whipsaw_rate"].median(), 0)})
        verdict.append({"規則": f"{k} {label}", "1. 回撤少 ≥15 個百分點": f"{'✅' if c1 else '❌'} {_pct(dd_gain)}",
                        "2. ≥60% 個股夏普勝過持有": f"{'✅' if c2 else '❌'} {_pct(share, 0)}",
                        "3. 兩段期間都過半": f"{'✅' if c3 else '❌'} " + " / ".join(_pct(v, 0) for v in halves.values()),
                        "可信": "✅" if (c1 and c2 and c3) else "❌"})
    summary, verdict = pd.DataFrame(summary), pd.DataFrame(verdict)
    print(summary.to_string(index=False), "\n")
    print(verdict.to_string(index=False), "\n")

    hold = res[(res["period"] == "全期") & res["ticker"].isin(HOLDINGS)].copy()
    print(hold[["ticker", "rule", "cagr", "max_dd", "sharpe", "time_defensive"]].round(3).to_string(index=False))
    names = {"BH": "一直持有", **{k: f"{k} {v}" for k, v in RULES.items()}}
    hold_html = tbl(pd.DataFrame({
        "股票": hold["ticker"], "規則": hold["rule"].map(names), "年化報酬": hold["cagr"].map(_pct),
        "最大回撤": hold["max_dd"].map(_pct), "夏普": hold["sharpe"].map(lambda x: f"{x:.2f}"),
        "空手時間": hold["time_defensive"].map(lambda x: _pct(x, 0))}))
    html = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>用指數訊號管理個股</title>
<style>
body{{background:#0D1117;color:#E6EDF3;font-family:-apple-system,"Segoe UI","Noto Sans TC",sans-serif;max-width:1150px;margin:0 auto;padding:16px;line-height:1.6}}
h1{{font-size:24px}} h2{{font-size:19px;margin-top:36px;border-bottom:1px solid #30363D;padding-bottom:6px}}
.tbl{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums;display:block;overflow-x:auto}}
.tbl th,.tbl td{{padding:6px 10px;border-bottom:1px solid #21262D;text-align:right;white-space:nowrap}}
.tbl th:first-child,.tbl td:first-child{{text-align:left}} .tbl th{{color:#8B949E;font-weight:600}}
.note{{color:#8B949E;font-size:13px}}
</style></head><body>
<h1>第二個假設：用指數的出場訊號管理高波動個股</h1>
<p class="note">第一輪（個股自己的出場規則）沒有規則通過。這是事先宣告的第二個、也是最後一個假設：個股只在指數趨勢規則說「持有」時才持有。
個股清單、交易成本（0.2%）、可信標準都和第一輪相同。仍有存活者偏誤。</p>
<h2>結論</h2>{tbl(verdict)}
<h2>各規則摘要（個股中位數）</h2>{tbl(summary)}
<h2>你的持股</h2>{hold_html}
<p class="note">歷史回測不保證未來，這不是投資建議。</p></body></html>"""
    REPORT_DIR.mkdir(exist_ok=True)
    out = REPORT_DIR / "market_exit_study.html"
    out.write_text(html, encoding="utf-8")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
