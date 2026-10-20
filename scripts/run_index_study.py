"""Run the same warning / entry / overlay rules on the S&P 500 and the Nasdaq-100.

Usage:
    python scripts/run_index_study.py [--refresh]
Writes reports/index_study.html.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bubble_monitor import daily_signals as ds  # noqa: E402
from bubble_monitor import overlay as ov  # noqa: E402
from bubble_monitor.backtest import find_drawdown_events  # noqa: E402
from bubble_monitor.data_sources import load_many  # noqa: E402

CACHE_DIR = ROOT / "data" / "backtest_cache"
REPORT_DIR = ROOT / "reports"
FRED_IDS = {"hy": "BAMLH0A0HYM2", "baa10y": "BAA10Y", "t10y2y": "T10Y2Y", "tbill": "DTB3"}
YAHOO_IDS = {"spx": "^GSPC", "ndx": "^NDX", "rut": "^RUT", "vix_daily": "^VIX", "vix3m_daily": "^VIX3M"}
INDEXES = {"spx": "S&P 500", "ndx": "那斯達克 100"}
START = "1990-01-01"
PERIODS = {"全期": ("1990-01-01", "2026-12-31"), "1990–2007": ("1990-01-01", "2007-12-31"),
           "2008–2026": ("2008-01-01", "2026-12-31")}
STRATS = {"buy_hold": "一直持有", "warn4": "警示 ≥4 減碼 50%", "faber": "10 個月均線",
          "combo_a": "組合 A：均線 × 警示", "combo_b": "組合 B：均線 × 警示 + 恐慌進場"}


def _pct(x, d=1):
    return "" if pd.isna(x) else f"{x * 100:.{d}f}%"


def tbl(df: pd.DataFrame) -> str:
    return df.to_html(index=False, escape=False, border=0, classes="tbl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 220)

    raw = load_many(FRED_IDS, YAHOO_IDS, CACHE_DIR, args.refresh)
    d = ds.build_daily_frame(raw)
    cash = raw["tbill"]

    warn_rows, entry_rows, perf_rows, bear_html, lead_html = [], [], [], "", ""
    sims_all = {}
    for key, name in INDEXES.items():
        px_full = d["spx"] if key == "spx" else raw[key].reindex(d.index).ffill(limit=2)
        sig_full = ds.compute_signals(d, None if key == "spx" else raw[key])
        fwd_full = ds.forward_outcomes(px_full.ffill())
        sig, fwd, px = sig_full.loc[START:], fwd_full.loc[START:], px_full.loc[START:].dropna()

        # warning hit rates (per episode, near the high)
        base = fwd["maxdd_126"][sig["near_high"]].dropna()
        for lvl in (3, 4):
            st = ds.warning_episode_stats(sig, fwd, lvl)
            warn_rows.append({"指數": name, "警示": f"≥{lvl} 個條件", "出現次數": st["episodes"],
                              "6 個月內跌 ≥10%": f"{st['hit10']}（{st['hit10'] / max(st['episodes'], 1):.0%}）",
                              "跌 ≥20%": f"{st['hit20']}（{st['hit20'] / max(st['episodes'], 1):.0%}）",
                              "基準：跌 ≥10%": _pct((base <= -0.10).mean(), 0),
                              "基準：跌 ≥20%": _pct((base <= -0.20).mean(), 0)})
        lead = ds.correction_lead_table(sig)
        caught = lead["lead_days_L3"].notna().sum()
        warn_rows[-1]["跌破 -10% 前 60 天有 ≥3 警告"] = f"{caught}/{len(lead)}"
        warn_rows[-2]["跌破 -10% 前 60 天有 ≥3 警告"] = f"{caught}/{len(lead)}"
        lead_html += f"<h3>{name}</h3>" + tbl(pd.DataFrame({
            "跌破 -10% 的日期": lead["crossed_-10%"], "最終跌幅": lead["max_decline"].map(_pct),
            "≥3 條件提前（交易日）": lead["lead_days_L3"].map(lambda x: "" if pd.isna(x) else f"{x:.0f}")}))

        # entry signals
        e_sum, _ = ds.entry_stats(sig, fwd, px_full.loc[START:].ffill())
        for _, r in e_sum.iterrows():
            entry_rows.append({"指數": name, "訊號": {"baseline": "基準：跌 15% 就買", "capitulation": "恐慌高峰",
                                                   "combo": "多重確認"}.get(r["signal"], r["signal"]),
                               "次數": r["count"], "12 個月後上漲": _pct(r["win_252"], 0),
                               "平均 12 個月報酬": _pct(r["avg_252"]), "之後平均再跌": _pct(r["avg_further_drop_126"]),
                               "最多再跌": _pct(r["worst_further_drop_126"])})

        # overlay strategies (price index + T-bill cash, same treatment for both indexes)
        entry = sig_full[["e_capitulation", "e_combo"]].fillna(False).astype(bool).any(axis=1)
        faber = ov.faber_exposure(px_full.dropna())
        warn4 = ov.warning_overlay(sig_full.dropna(subset=["dd_1y"]), 4)
        exp = {"buy_hold": pd.Series(1.0, index=px.index), "warn4": warn4, "faber": faber,
               "combo_a": ov.combined_exposure(faber, warn4), "combo_b": ov.combined_exposure(faber, warn4, entry)}
        sims = {k: ov.simulate(px, e, cash) for k, e in exp.items()}
        sims_all[key] = sims
        for pname, (s, e) in PERIODS.items():
            for k, sim in sims.items():
                sub = sim.loc[s:e].copy()
                sub["equity"] = (1 + sub["ret"]).cumprod()
                m = ov.metrics(sub, cash)
                perf_rows.append({"index": name, "period": pname, "strategy": k, **m})
        bears = [b for b in find_drawdown_events(px, 0.20) if b.peak_date >= pd.Timestamp(START)]
        bt_ = ov.bear_table(px, sims, bears)
        bear_html += f"<h3>{name}</h3>" + tbl(pd.DataFrame({
            "空頭高點": bt_["peak"], "低點": bt_["trough"], "指數跌幅": bt_["index_drop"].map(_pct),
            **{STRATS[k]: bt_[k].map(_pct) for k in STRATS}}))
        print(name)
        print(bt_.round(3).to_string(index=False), "\n")

    warn = pd.DataFrame(warn_rows)
    entry = pd.DataFrame(entry_rows)
    perf = pd.DataFrame(perf_rows)
    print(warn.to_string(index=False), "\n")
    print(entry.to_string(index=False), "\n")
    print(perf.round(3).to_string(index=False))

    perf_html = ""
    for pname in PERIODS:
        p = perf[perf["period"] == pname]
        perf_html += f"<h3>{pname}</h3>" + tbl(pd.DataFrame({
            "指數": p["index"], "策略": p["strategy"].map(STRATS), "年化報酬": p["cagr"].map(_pct),
            "最大回撤": p["max_dd"].map(_pct), "夏普": p["sharpe"].map(lambda x: f"{x:.2f}"),
            "報酬 ÷ 最大回撤": p["calmar"].map(lambda x: f"{x:.2f}"), "減碼時間": p["time_defensive"].map(lambda x: _pct(x, 0)),
        }))

    fig = make_subplots(rows=1, cols=2, subplot_titles=list(INDEXES.values()))
    colors = {"buy_hold": "#8B949E", "warn4": "#F0883E", "faber": "#58A6FF", "combo_a": "#BC8CFF", "combo_b": "#2EA043"}
    for col, key in enumerate(INDEXES, start=1):
        for k, sim in sims_all[key].items():
            fig.add_trace(go.Scatter(x=sim.index, y=sim["equity"], name=STRATS[k], legendgroup=k, showlegend=col == 1,
                                     line=dict(color=colors[k], width=1.3)), row=1, col=col)
        fig.update_yaxes(type="log", row=1, col=col)
    fig.update_layout(height=480, template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
                      margin=dict(l=50, r=20, t=40, b=30), legend=dict(orientation="h", y=-0.12))

    html = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>那斯達克對照</title>
<style>
body{{background:#0D1117;color:#E6EDF3;font-family:-apple-system,"Segoe UI","Noto Sans TC",sans-serif;max-width:1100px;margin:0 auto;padding:16px;line-height:1.6}}
h1{{font-size:24px}} h2{{font-size:19px;margin-top:36px;border-bottom:1px solid #30363D;padding-bottom:6px}} h3{{font-size:15px;margin-top:20px}}
.tbl{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums;display:block;overflow-x:auto}}
.tbl th,.tbl td{{padding:6px 10px;border-bottom:1px solid #21262D;text-align:right;white-space:nowrap}}
.tbl th:first-child,.tbl td:first-child,.tbl td:nth-child(2){{text-align:left}} .tbl th{{color:#8B949E;font-weight:600}}
.note{{color:#8B949E;font-size:13px}}
</style></head><body>
<h1>同一套規則：S&amp;P 500 vs 那斯達克 100（1990–2026）</h1>
<p class="note">規則完全不變，只把「距高點、50 日均線、跌幅門檻、10 個月均線」換成目標指數；信用利差、VIX、小型股廣度、殖利率曲線是全市場指標，兩邊共用。
兩個指數都用價格指數（不含股息）、現金賺國庫券利率，方便直接比較。</p>
<h2>資金曲線</h2>{fig.to_html(full_html=False, include_plotlyjs="cdn")}
<h2>減碼策略績效</h2>{perf_html}
<h2>每次空頭期間的虧損</h2>{bear_html}
<h2>早期警示命中率（以次計，接近高點時）</h2>{tbl(warn)}
<h3>每次跌破 -10% 前的警告</h3>{lead_html}
<h2>進場訊號</h2>{tbl(entry)}
<p class="note">歷史回測不保證未來，這不是投資建議。</p>
</body></html>"""
    REPORT_DIR.mkdir(exist_ok=True)
    out = REPORT_DIR / "index_study.html"
    out.write_text(html, encoding="utf-8")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
