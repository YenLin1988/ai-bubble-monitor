"""Does acting on the warning / entry signals reduce losses? US test plus foreign-market validation.

Usage:
    python scripts/run_overlay_study.py [--refresh]
Writes reports/overlay_study.html.
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
YAHOO_IDS = {"spx": "^GSPC", "spx_tr": "^SP500TR", "rut": "^RUT", "vix_daily": "^VIX", "vix3m_daily": "^VIX3M"}
FOREIGN = {"^N225": "日經 225", "^HSI": "恆生", "^GDAXI": "德國 DAX", "^FTSE": "英國 FTSE",
           "^TWII": "台灣加權", "^KS11": "韓國 KOSPI", "^AXJO": "澳洲 ASX200", "^GSPTSE": "加拿大 TSX"}
US_START = "1990-01-01"
PERIODS = {"全期": ("1990-01-01", "2026-12-31"), "1990–2007": ("1990-01-01", "2007-12-31"),
           "2008–2026": ("2008-01-01", "2026-12-31")}
NAMES = {"buy_hold": "一直持有", "warn4": "警示 ≥4 減碼 50%", "warn3": "警示 ≥3 減碼 50%",
         "faber": "10 個月均線（全進全出）", "price_only": "純價格版減碼 50%", "us_warn4": "用美國警示 ≥4 減碼 50%",
         "combo_a": "組合 A：均線 × 警示", "combo_b": "組合 B：均線 × 警示 + 恐慌進場"}
ENTRY_COLS = ("e_capitulation", "e_combo")


def _pct(x, d=1):
    return "" if pd.isna(x) else f"{x * 100:.{d}f}%"


def tbl(df: pd.DataFrame) -> str:
    return df.to_html(index=False, escape=False, border=0, classes="tbl")


def run_set(price: pd.Series, exposures: dict[str, pd.Series], cash) -> dict[str, pd.DataFrame]:
    return {k: ov.simulate(price, e, cash) for k, e in exposures.items()}


def metric_rows(sims: dict, cash, start: str, end: str) -> pd.DataFrame:
    rows = []
    for k, sim in sims.items():
        sub = sim.loc[start:end].copy()
        if len(sub) < 250:
            continue
        sub["equity"] = (1 + sub["ret"]).cumprod()
        m = ov.metrics(sub, cash)
        rows.append({"strategy": k, **m})
    return pd.DataFrame(rows)


def fmt_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "策略": df["strategy"].map(NAMES), "年化報酬": df["cagr"].map(_pct), "波動": df["vol"].map(_pct),
        "最大回撤": df["max_dd"].map(_pct), "夏普": df["sharpe"].map(lambda x: f"{x:.2f}"),
        "報酬 ÷ 最大回撤": df["calmar"].map(lambda x: f"{x:.2f}"),
        "減碼時間": df["time_defensive"].map(lambda x: _pct(x, 0)), "交易次數": df["trades"],
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 220)

    raw = load_many(FRED_IDS, YAHOO_IDS, CACHE_DIR, args.refresh)
    cash = raw["tbill"]
    sig = ds.compute_signals(ds.build_daily_frame(raw))
    tr = raw["spx_tr"].loc[US_START:]
    spx = raw["spx"]

    # ---------------- US ----------------
    ps = ov.price_only_signals(spx)
    us_exp = {
        "buy_hold": pd.Series(1.0, index=tr.index),
        "warn4": ov.warning_overlay(sig, 4),
        "warn3": ov.warning_overlay(sig, 3),
        "faber": ov.faber_exposure(spx),
        "price_only": ov.warning_overlay(ps, 2, entry_cols=("e_entry",)),
    }
    us_entry = sig[list(ENTRY_COLS)].any(axis=1)
    us_exp["combo_a"] = ov.combined_exposure(us_exp["faber"], us_exp["warn4"])
    us_exp["combo_b"] = ov.combined_exposure(us_exp["faber"], us_exp["warn4"], us_entry)
    us_sims = run_set(tr, us_exp, cash)
    us_tables = {p: metric_rows(us_sims, cash, *rng) for p, rng in PERIODS.items()}
    bears = [e for e in find_drawdown_events(spx, 0.20) if e.peak_date >= pd.Timestamp(US_START)]
    corr_events = [e for e in find_drawdown_events(spx, 0.10) if e.peak_date >= pd.Timestamp(US_START)]
    us_bear = ov.bear_table(tr, us_sims, bears)
    # cost side: return given up during non-bear periods = full-sample CAGR gap is the net answer
    for p, t in us_tables.items():
        print(f"== US {p} ==")
        print(t.round(3).to_string(index=False), "\n")
    print(us_bear.round(3).to_string(index=False), "\n")

    # ---------------- Foreign validation ----------------
    foreign_rows, foreign_bears = [], []
    for ticker, name in FOREIGN.items():
        px = load_many({}, {ticker.strip("^"): ticker}, CACHE_DIR, args.refresh)[ticker.strip("^")]
        px = px[px > 0]
        # US signals as of the previous calendar day (the US closes after Asia/Europe).
        us_lagged = sig.copy()
        us_lagged.index = us_lagged.index + pd.Timedelta(days=1)
        us_on_local = us_lagged.reindex(us_lagged.index.union(px.index)).ffill().reindex(px.index)
        us_on_local = us_on_local.dropna(subset=["warning_level"])
        ps_local = ov.price_only_signals(px)
        exp = {
            "buy_hold": pd.Series(1.0, index=px.index),
            "us_warn4": ov.warning_overlay(us_on_local.astype({"near_high": bool}), 4),
            "faber": ov.faber_exposure(px),
            "price_only": ov.warning_overlay(ps_local, 2, entry_cols=("e_entry",)),
        }
        local_entry = us_on_local[list(ENTRY_COLS)].fillna(False).astype(bool).any(axis=1)
        exp["combo_a"] = ov.combined_exposure(exp["faber"], exp["us_warn4"])
        exp["combo_b"] = ov.combined_exposure(exp["faber"], exp["us_warn4"], local_entry)
        sims = run_set(px, exp, None)
        local_bears = find_drawdown_events(px, 0.20)
        for label, start in (("全期", px.index[0] + pd.DateOffset(years=1)), ("1990 後", pd.Timestamp("1990-01-01"))):
            start = max(start, px.index[0] + pd.DateOffset(years=1))
            m = metric_rows(sims, None, str(start.date()), "2026-12-31")
            for _, r in m.iterrows():
                foreign_rows.append({"market": name, "period": label, "since": start.year, **r.to_dict()})
        bt = ov.bear_table(px, sims, [e for e in local_bears if e.peak_date >= px.index[0] + pd.DateOffset(years=1)])
        bt.insert(0, "market", name)
        foreign_bears.append(bt)
        print(name, "done", len(local_bears), "bears")
    fr = pd.DataFrame(foreign_rows)
    fb = pd.concat(foreign_bears, ignore_index=True)

    # summary: per strategy, how many markets improved vs buy & hold (full period)
    full = fr[fr["period"] == "全期"]
    summary = []
    for k in ["us_warn4", "faber", "price_only", "combo_a", "combo_b"]:
        diffs = []
        for mkt, g in full.groupby("market"):
            bh = g[g["strategy"] == "buy_hold"].iloc[0]
            s = g[g["strategy"] == k]
            if not len(s):
                continue
            s = s.iloc[0]
            diffs.append({"cagr": s["cagr"] - bh["cagr"], "mdd": s["max_dd"] - bh["max_dd"],
                          "calmar_better": s["calmar"] > bh["calmar"], "sharpe_better": s["sharpe"] > bh["sharpe"]})
        d = pd.DataFrame(diffs)
        bear_loss = fb[k] - fb["buy_hold"] if k in fb else pd.Series(dtype=float)
        summary.append({
            "策略": NAMES[k], "市場數": len(d),
            "年化報酬差（中位數）": _pct(d["cagr"].median(), 2),
            "最大回撤改善（中位數）": _pct(d["mdd"].median(), 1),
            "報酬÷回撤 變好的市場": f"{int(d['calmar_better'].sum())}/{len(d)}",
            "夏普變好的市場": f"{int(d['sharpe_better'].sum())}/{len(d)}",
            "空頭期間少虧（中位數）": _pct(bear_loss.median(), 1),
            "空頭少虧的比例": f"{int((bear_loss > 0).sum())}/{int(bear_loss.notna().sum())}",
        })
    summary = pd.DataFrame(summary)
    print(summary.to_string(index=False))
    print(fb.round(3).to_string(index=False))

    # ---------------- report ----------------
    fig = go.Figure()
    colors = {"buy_hold": "#8B949E", "warn4": "#F0883E", "warn3": "#D29922", "faber": "#58A6FF", "price_only": "#39D2C0",
              "combo_a": "#BC8CFF", "combo_b": "#2EA043"}
    for k, sim in us_sims.items():
        fig.add_trace(go.Scatter(x=sim.index, y=sim["equity"], name=NAMES[k], line=dict(color=colors[k], width=1.4)))
    for e in bears:
        fig.add_vrect(x0=e.peak_date, x1=e.trough_date, fillcolor="rgba(248,81,73,0.12)", line_width=0)
    fig.update_yaxes(type="log", title_text="1 元成長到（對數）")
    fig.update_layout(height=520, template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
                      margin=dict(l=60, r=20, t=30, b=30), legend=dict(orientation="h", y=1.08))

    bear_fmt = pd.DataFrame({"空頭高點": us_bear["peak"], "低點": us_bear["trough"],
                             **{NAMES[k]: us_bear[k].map(_pct) for k in us_sims}})
    us_html = "".join(f"<h3>{p}</h3>{tbl(fmt_metrics(t))}" for p, t in us_tables.items())
    fr_fmt = fr[fr["period"] == "全期"].copy()
    fr_html = tbl(pd.DataFrame({
        "市場": fr_fmt["market"], "起始": fr_fmt["since"], "策略": fr_fmt["strategy"].map(NAMES),
        "年化報酬": fr_fmt["cagr"].map(_pct), "最大回撤": fr_fmt["max_dd"].map(_pct),
        "報酬 ÷ 最大回撤": fr_fmt["calmar"].map(lambda x: f"{x:.2f}"), "減碼時間": fr_fmt["time_defensive"].map(lambda x: _pct(x, 0)),
    }))
    fb_fmt = pd.DataFrame({"市場": fb["market"], "空頭高點": fb["peak"], "指數跌幅": fb["index_drop"].map(_pct),
                           **{NAMES[k]: fb[k].map(_pct) for k in ["buy_hold", "us_warn4", "faber", "price_only", "combo_a", "combo_b"] if k in fb}})
    html = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>減碼策略回測</title>
<style>
body{{background:#0D1117;color:#E6EDF3;font-family:-apple-system,"Segoe UI","Noto Sans TC",sans-serif;max-width:1100px;margin:0 auto;padding:16px;line-height:1.6}}
h1{{font-size:24px}} h2{{font-size:19px;margin-top:36px;border-bottom:1px solid #30363D;padding-bottom:6px}} h3{{font-size:15px;margin-top:20px}}
.tbl{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums;display:block;overflow-x:auto}}
.tbl th,.tbl td{{padding:6px 10px;border-bottom:1px solid #21262D;text-align:right;white-space:nowrap}}
.tbl th:first-child,.tbl td:first-child{{text-align:left}} .tbl th{{color:#8B949E;font-weight:600}}
.note{{color:#8B949E;font-size:13px}}
</style></head><body>
<h1>照訊號減碼，能不能少賠？</h1>
<p class="note">規則測試前就固定。訊號收盤確認、隔天收盤交易；每次調整部位扣 0.1% 成本。美國用含股息的 S&amp;P 500 總報酬指數，現金賺國庫券利率。</p>
<h2>美國 1990–2026：資金曲線</h2>{fig.to_html(full_html=False, include_plotlyjs="cdn")}
<h2>美國：績效</h2>{us_html}
<h2>美國：每次空頭期間的虧損</h2>{tbl(bear_fmt)}
<h2>海外驗證：總結</h2>
<p class="note">海外指數為價格指數（不含股息），減碼時現金不計利息，對減碼策略較不利。「用美國警示」= 直接拿美國的早期警示操作當地市場；「純價格版」只用當地市場自己的價格。</p>
{tbl(summary)}
<h2>海外驗證：各市場</h2>{fr_html}
<h2>海外驗證：各市場空頭期間的虧損</h2>{tbl(fb_fmt)}
<h2>規則</h2><ul class="note">
<li>警示減碼：大盤距一年高點 10% 內、早期警示達門檻 → 部位 50%；出現進場訊號（恐慌高峰或多重確認），或警示連續 5 天 ≤1 → 回到 100%。</li>
<li>10 個月均線（Faber 2007）：月底收盤低於 10 個月均線 → 全部現金，否則 100%。</li>
<li>純價格版：跌破 50 日均線且 20 日實現波動 ≥ 其 50 日均值 1.3 倍 → 50%；跌 ≥15%、實現波動曾 ≥30% 且回落 20%，或兩個條件連續 5 天都消失 → 100%。</li>
</ul><p class="note">歷史回測不保證未來，這不是投資建議。</p>
</body></html>"""
    REPORT_DIR.mkdir(exist_ok=True)
    out = REPORT_DIR / "overlay_study.html"
    out.write_text(html, encoding="utf-8")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
