"""Compare candidate bottom signals on two separate periods and write reports/bottom_study.html.

Usage:
    python scripts/run_bottom_study.py [--refresh]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bubble_monitor import bottom_signals as bs  # noqa: E402
from bubble_monitor import daily_signals as ds  # noqa: E402
from bubble_monitor.data_sources import load_many  # noqa: E402

CACHE_DIR = ROOT / "data" / "backtest_cache"
REPORT_DIR = ROOT / "reports"
FRED_IDS = {"hy": "BAMLH0A0HYM2", "baa10y": "BAA10Y", "t10y2y": "T10Y2Y"}
YAHOO_IDS = {"spx": "^GSPC", "rut": "^RUT", "vix_daily": "^VIX", "vix3m_daily": "^VIX3M"}
PERIODS = {"全期 1990–2026": ("1990-01-01", "2026-12-31"),
           "前段 1990–2007": ("1990-01-01", "2007-12-31"),
           "後段 2008–2026": ("2008-01-01", "2026-12-31")}


def _pct(x, d=0):
    return "" if pd.isna(x) else f"{x * 100:.{d}f}%"


def tbl(df: pd.DataFrame) -> str:
    return df.to_html(index=False, escape=False, border=0, classes="tbl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    raw = load_many(FRED_IDS, YAHOO_IDS, CACHE_DIR, args.refresh)
    d_full = ds.build_daily_frame(raw)
    s_full = ds.compute_signals(d_full)
    fwd_full = ds.forward_outcomes(d_full["spx"])

    signals = {"baseline": ("基準：第一次跌破 -15% 就買", "大盤距一年高點第一次跌超過 15% 的那天",
                            s_full["dd_1y"] <= -0.15)}
    for key, (label, desc, fn) in bs.CANDIDATES.items():
        signals[key] = (label, desc, fn(d_full, s_full))

    results, episodes = {}, {}
    for pname, (start, end) in PERIODS.items():
        d = d_full.loc[start:end]
        s, fwd, spx = s_full.loc[start:end], fwd_full.loc[start:end], d["spx"]
        troughs = bs.decline_troughs(spx, s["dd_1y"])
        for key, (label, _, sig) in signals.items():
            summary, eps = bs.evaluate_candidate(sig.loc[start:end], spx, fwd, troughs)
            results[(pname, key)] = summary
            if pname.startswith("全期"):
                episodes[key] = eps
        if pname.startswith("全期"):
            all_troughs = troughs

    pd.set_option("display.width", 220)
    print(all_troughs.to_string(index=False), "\n")
    for pname in PERIODS:
        df = pd.DataFrame({k: results[(pname, k)] for k in signals}).T
        print(f"== {pname} ==")
        print(df.to_string(), "\n")
    for k, eps in episodes.items():
        print(k)
        print(eps.round(3).to_string(index=False), "\n")

    # ---- report ----
    sections = ""
    for pname in PERIODS:
        rows = []
        for k, (label, _, _) in signals.items():
            r = results[(pname, k)]
            rows.append({
                "方法": label, "訊號次數": r["signals"],
                "12 個月後上漲": _pct(r["win_252"]), "平均 12 個月報酬": _pct(r["avg_252"], 1),
                "中位數": _pct(r["median_252"], 1), "最差 12 個月": _pct(r["worst_252"], 1),
                "訊號後平均再跌": _pct(r["avg_further_drop"], 1), "最多再跌": _pct(r["worst_further_drop"], 1),
                "離真正低點 30 天內": _pct(r["near_trough"]), "抓到的底部": r["troughs_caught"],
            })
        sections += f"<h2>{pname}</h2>{tbl(pd.DataFrame(rows))}"

    spx = d_full["spx"].loc["1990":]
    fig = go.Figure(go.Scatter(x=spx.index, y=spx.values, name="S&P 500", line=dict(color="#58A6FF", width=1.1)))
    palette = ["#8B949E", "#2EA043", "#D29922", "#BC8CFF", "#F0883E", "#39D2C0", "#F85149"]
    for (k, (label, _, _)), color in zip(signals.items(), palette):
        dates = episodes[k]["date"] if len(episodes[k]) else []
        fig.add_trace(go.Scatter(x=dates, y=spx.reindex(dates).values, mode="markers", name=label,
                                 marker=dict(color=color, size=8, line=dict(color="#0D1117", width=1))))
    for t in all_troughs["trough"]:
        fig.add_vline(x=t, line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot"))
    fig.update_yaxes(type="log")
    fig.update_layout(height=560, template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
                      margin=dict(l=50, r=20, t=30, b=30), legend=dict(orientation="h", y=1.08))

    rules = "".join(f"<li><b>{label}</b>：{desc}</li>" for label, desc, _ in signals.values())
    troughs_fmt = pd.DataFrame({"跌破 -15%": all_troughs["crossed"].dt.date, "真正低點": all_troughs["trough"].dt.date,
                                "最大跌幅": all_troughs["depth"].map(_pct)})
    html = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>底部訊號比較</title>
<style>
body{{background:#0D1117;color:#E6EDF3;font-family:-apple-system,"Segoe UI","Noto Sans TC",sans-serif;max-width:1100px;margin:0 auto;padding:16px;line-height:1.6}}
h1{{font-size:24px}} h2{{font-size:19px;margin-top:36px;border-bottom:1px solid #30363D;padding-bottom:6px}}
.tbl{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums;display:block;overflow-x:auto}}
.tbl th,.tbl td{{padding:6px 10px;border-bottom:1px solid #21262D;text-align:right;white-space:nowrap}}
.tbl th:first-child,.tbl td:first-child{{text-align:left}} .tbl th{{color:#8B949E;font-weight:600}}
.note{{color:#8B949E;font-size:13px}}
</style></head><body>
<h1>底部訊號比較（1990–2026）</h1>
<p class="note">每個方法都用公開的原始參數，或在測試前就決定的規則，沒有針對歷史調整。分前後兩段檢驗，要兩段都好才算真的有效。
「抓到的底部」= 跌幅 ≥15% 的修正中，有訊號落在真正低點前後 30 個交易日內的比例。</p>
<h2>訊號位置</h2><p class="note">虛線 = 每次修正的真正低點。</p>{fig.to_html(full_html=False, include_plotlyjs="cdn")}
{sections}
<h2>修正低點清單（跌 ≥15%）</h2>{tbl(troughs_fmt)}
<h2>規則</h2><ul class="note">{rules}</ul>
<h2>沒有測試的方法</h2><ul class="note">
<li>Zweig 廣度推力、Lowry 90% 日：需要紐約證交所漲跌家數與上漲/下跌成交量的歷史資料，沒有免費來源。</li>
<li>AAII 散戶情緒、Put/Call 比率：免費歷史資料不完整。</li></ul>
<h2>限制</h2><ul class="note"><li>跌 ≥15% 的修正只有十多次，每個方法的訊號次數都很少，統計誤差很大。</li>
<li>這是歷史統計，不是投資建議。</li></ul>
</body></html>"""
    REPORT_DIR.mkdir(exist_ok=True)
    out = REPORT_DIR / "bottom_study.html"
    out.write_text(html, encoding="utf-8")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
