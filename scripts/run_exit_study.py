"""Which exit (de-risking) rule is credible for the Philadelphia Semiconductor index? Pre-registered test.

Candidates (published rules, original parameters), each paired with the same re-entry overlay
(after a capitulation / multi-confirmation entry signal hold at least 50% until the rule is back in):
    E1 Faber 10-month average (current)          E4 20% trailing stop, re-enter above 200-day
    E2 200-day average, daily                     E5 12-month absolute momentum vs T-bills
    E3 50/200 golden / death cross                E6 volatility targeting, 25% / 20-day vol

Credibility criteria (all must hold):
    1. SOX: Sharpe and max drawdown better than buy-and-hold in 1994-2007 AND 2008-2026
    2. Sharpe better than buy-and-hold in >= 4 of 5 validation markets (NDX, TWII, KS11, N225, SMH)
    3. SOX full-period Calmar above E1, and Sharpe not below E1 in a majority of validation markets

Usage:
    python scripts/run_exit_study.py [--refresh]
Writes reports/exit_study.html.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bubble_monitor import daily_signals as ds  # noqa: E402
from bubble_monitor import overlay as ov  # noqa: E402
from bubble_monitor.backtest import find_drawdown_events  # noqa: E402
from bubble_monitor.data_sources import load_many  # noqa: E402

CACHE_DIR = ROOT / "data" / "backtest_cache"
REPORT_DIR = ROOT / "reports"
FRED_IDS = {"hy": "BAMLH0A0HYM2", "baa10y": "BAA10Y", "t10y2y": "T10Y2Y", "tbill": "DTB3"}
YAHOO_IDS = {"spx": "^GSPC", "rut": "^RUT", "vix_daily": "^VIX", "vix3m_daily": "^VIX3M", "sox": "^SOX",
             "ndx": "^NDX", "TWII": "^TWII", "KS11": "^KS11", "N225": "^N225", "smh": "SMH"}
VALIDATION = {"ndx": ("那斯達克 100", True), "TWII": ("台灣加權", False), "KS11": ("韓國 KOSPI", False),
              "N225": ("日經 225", False), "smh": ("半導體 ETF SMH", True)}  # bool: US-listed (earns T-bill cash)
HALVES = {"1994–2007": ("1994-01-01", "2007-12-31"), "2008–2026": ("2008-01-01", "2026-12-31")}
RULES = {"E1": "10 個月均線（現行）", "E2": "200 日均線", "E3": "死亡／黃金交叉", "E4": "20% 移動停損",
         "E5": "12 個月絕對動能", "E6": "波動率目標 25%"}


def _pct(x, d=1):
    return "" if pd.isna(x) else f"{x * 100:.{d}f}%"


def tbl(df: pd.DataFrame) -> str:
    return df.to_html(index=False, escape=False, border=0, classes="tbl")


def rule_exposures(px: pd.Series, cash) -> dict[str, pd.Series]:
    return {"E1": ov.faber_exposure(px), "E2": ov.sma_exposure(px, 200), "E3": ov.cross_exposure(px, 50, 200),
            "E4": ov.trailing_stop_exposure(px, 0.20, 200), "E5": ov.abs_momentum_exposure(px, cash),
            "E6": ov.vol_target_exposure(px, 0.25, 20)}


def entry_flags(d: pd.DataFrame, raw_px: pd.Series, local_index: pd.DatetimeIndex) -> pd.Series:
    """Capitulation / multi-confirmation entry for this market, known after the US close, used next local day."""
    sig = ds.compute_signals(d, raw_px)
    e = sig[["e_capitulation", "e_combo"]].fillna(False).astype(bool).any(axis=1)
    e.index = e.index + pd.Timedelta(days=1)
    return e.reindex(e.index.union(local_index)).fillna(False).astype(bool).reindex(local_index).fillna(False)


def run_market(px: pd.Series, entry: pd.Series, cash, start: str) -> dict[str, pd.DataFrame]:
    p = px.loc[start:].dropna()
    exps = rule_exposures(px.dropna(), cash)
    sims = {"buy_hold": ov.simulate(p, pd.Series(1.0, index=p.index), cash)}
    for k, e in exps.items():
        sims[k] = ov.simulate(p, ov.combined_exposure(e, pd.Series(1.0, index=e.index), entry), cash)
    return sims


def period_metrics(sim: pd.DataFrame, cash, a: str, b: str) -> dict:
    sub = sim.loc[a:b].copy()
    sub["equity"] = (1 + sub["ret"]).cumprod()
    return ov.metrics(sub, cash)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 230)

    raw = load_many(FRED_IDS, YAHOO_IDS, CACHE_DIR, args.refresh)
    cash = raw["tbill"]
    d = ds.build_daily_frame(raw)

    # ---------------- SOX ----------------
    sox = raw["sox"].dropna()
    sox_start = str((sox.index[0] + pd.DateOffset(years=1)).date())  # 1-year warm-up for 200-day rules
    sox_sims = run_market(sox, entry_flags(d, sox, sox.index), cash, sox_start)
    rows = []
    for name, (a, b) in {"全期": (sox_start, "2026-12-31"), **HALVES}.items():
        for k, sim in sox_sims.items():
            rows.append({"period": name, "rule": k, **period_metrics(sim, cash, a, b)})
    sox_tbl = pd.DataFrame(rows)
    print(sox_tbl.round(3).to_string(index=False), "\n")

    bears = [e for e in find_drawdown_events(sox.loc[sox_start:], 0.20)]
    bear = ov.bear_table(sox, sox_sims, bears)
    # how late does each rule exit? index drawdown from the peak on the first day exposure drops below 1
    late_rows = []
    for ev in bears:
        row = {"peak": ev.peak_date.date(), "drop": ev.depth}
        for k in RULES:
            exp = sox_sims[k]["exposure"].loc[ev.peak_date:ev.trough_date]
            cut = exp[exp < 1]
            row[k] = sox.asof(cut.index[0]) / sox.asof(ev.peak_date) - 1 if len(cut) else np.nan
        late_rows.append(row)
    late = pd.DataFrame(late_rows)
    print(bear.round(3).to_string(index=False), "\n")
    print(late.round(3).to_string(index=False), "\n")

    # ---------------- validation ----------------
    val_rows = []
    for key, (name, us) in VALIDATION.items():
        px = raw[key].dropna()
        px = px[px > 0]
        start = str((px.index[0] + pd.DateOffset(years=1)).date())
        c = cash if us else None
        sims = run_market(px, entry_flags(d, px, px.index), c, start)
        for k, sim in sims.items():
            val_rows.append({"market": name, "since": start[:4], "rule": k, **period_metrics(sim, c, start, "2026-12-31")})
    val = pd.DataFrame(val_rows)
    print(val.round(3).to_string(index=False), "\n")

    # ---------------- verdict ----------------
    def m(period, rule, col):
        return float(sox_tbl[(sox_tbl["period"] == period) & (sox_tbl["rule"] == rule)][col].iloc[0])

    verdict = []
    for k, label in RULES.items():
        c1 = all(m(h, k, "sharpe") > m(h, "buy_hold", "sharpe") and m(h, k, "max_dd") > m(h, "buy_hold", "max_dd")
                 for h in HALVES)
        better_bh = sum(float(g[g["rule"] == k]["sharpe"].iloc[0]) > float(g[g["rule"] == "buy_hold"]["sharpe"].iloc[0])
                        for _, g in val.groupby("market"))
        c2 = better_bh >= 4
        not_worse_e1 = sum(float(g[g["rule"] == k]["sharpe"].iloc[0]) >= float(g[g["rule"] == "E1"]["sharpe"].iloc[0]) - 1e-9
                           for _, g in val.groupby("market"))
        c3 = k == "E1" or (m("全期", k, "calmar") > m("全期", "E1", "calmar") and not_worse_e1 >= 3)
        verdict.append({"規則": f"{k} {label}", "1. SOX 兩段都勝過持有": "✅" if c1 else "❌",
                        "2. 驗證市場夏普勝過持有": f"{'✅' if c2 else '❌'} {better_bh}/5",
                        "3. 勝過現行 E1": "（基準）" if k == "E1" else f"{'✅' if c3 else '❌'} 驗證市場 {not_worse_e1}/5 不輸",
                        "可信": "✅" if (c1 and c2 and c3) else "❌"})
    verdict = pd.DataFrame(verdict)
    print(verdict.to_string(index=False))

    # ---------------- report ----------------
    names = {"buy_hold": "一直持有", **{k: f"{k} {v}" for k, v in RULES.items()}}
    sox_html = ""
    for period in ["全期", *HALVES]:
        t = sox_tbl[sox_tbl["period"] == period]
        sox_html += f"<h3>{period}</h3>" + tbl(pd.DataFrame({
            "規則": t["rule"].map(names), "年化報酬": t["cagr"].map(_pct), "最大回撤": t["max_dd"].map(_pct),
            "夏普": t["sharpe"].map(lambda x: f"{x:.2f}"), "報酬 ÷ 回撤": t["calmar"].map(lambda x: f"{x:.2f}"),
            "空手時間": t["time_defensive"].map(lambda x: _pct(x, 0)), "交易次數": t["trades"]}))
    val_html = tbl(pd.DataFrame({
        "市場": val["market"], "起始": val["since"], "規則": val["rule"].map(names), "年化報酬": val["cagr"].map(_pct),
        "最大回撤": val["max_dd"].map(_pct), "夏普": val["sharpe"].map(lambda x: f"{x:.2f}")}))
    bear_html = tbl(pd.DataFrame({"空頭高點": bear["peak"], "低點": bear["trough"], "SOX 跌幅": bear["index_drop"].map(_pct),
                                  **{names[k]: bear[k].map(_pct) for k in ["buy_hold", *RULES]}}))
    late_html = tbl(pd.DataFrame({"空頭高點": late["peak"], "最終跌幅": late["drop"].map(_pct),
                                  **{names[k]: late[k].map(_pct) for k in RULES}}))
    fig = go.Figure()
    palette = ["#8B949E", "#58A6FF", "#F0883E", "#BC8CFF", "#2EA043", "#D29922", "#39D2C0"]
    for (k, sim), c in zip(sox_sims.items(), palette):
        fig.add_trace(go.Scatter(x=sim.index, y=sim["equity"], name=names[k], line=dict(color=c, width=1.3)))
    fig.update_yaxes(type="log")
    fig.update_layout(height=500, template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
                      margin=dict(l=50, r=20, t=30, b=30), legend=dict(orientation="h", y=1.1))
    html = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>出場訊號研究</title>
<style>
body{{background:#0D1117;color:#E6EDF3;font-family:-apple-system,"Segoe UI","Noto Sans TC",sans-serif;max-width:1150px;margin:0 auto;padding:16px;line-height:1.6}}
h1{{font-size:24px}} h2{{font-size:19px;margin-top:36px;border-bottom:1px solid #30363D;padding-bottom:6px}} h3{{font-size:15px;margin-top:20px}}
.tbl{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums;display:block;overflow-x:auto}}
.tbl th,.tbl td{{padding:6px 10px;border-bottom:1px solid #21262D;text-align:right;white-space:nowrap}}
.tbl th:first-child,.tbl td:first-child,.tbl td:nth-child(3){{text-align:left}} .tbl th{{color:#8B949E;font-weight:600}}
.note{{color:#8B949E;font-size:13px}}
</style></head><body>
<h1>費城半導體 SOX：哪個出場訊號可信？</h1>
<p class="note">六個公開的經典規則，用原始參數，測試前就固定；每個規則都搭配同樣的恐慌進場（出現恐慌高峰／多重確認時先買回 50%）。
訊號收盤確認、隔天收盤交易，每次調整扣 0.1% 成本；美國市場空手時賺國庫券利率，海外市場不計利息。</p>
<h2>結論（標準在測試前固定）</h2>{tbl(verdict)}
<h2>SOX 資金曲線</h2>{fig.to_html(full_html=False, include_plotlyjs="cdn")}
<h2>SOX 績效</h2>{sox_html}
<h2>SOX 每次空頭期間的虧損</h2>{bear_html}
<h2>出場有多晚：開始減碼時，SOX 已經從高點跌了多少</h2>
<p class="note">空白 = 這次空頭期間沒有減碼。出場訊號一定會晚於最高點，這張表顯示晚多少。</p>{late_html}
<h2>驗證市場</h2>{val_html}
<p class="note">歷史回測不保證未來，這不是投資建議。</p></body></html>"""
    REPORT_DIR.mkdir(exist_ok=True)
    out = REPORT_DIR / "exit_study.html"
    out.write_text(html, encoding="utf-8")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
