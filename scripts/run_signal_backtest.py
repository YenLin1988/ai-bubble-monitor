"""Backtest the daily early-warning and entry signals and write reports/signals_report.html.

Usage:
    python scripts/run_signal_backtest.py [--refresh]
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
from bubble_monitor.data_sources import load_many  # noqa: E402

CACHE_DIR = ROOT / "data" / "backtest_cache"
REPORT_DIR = ROOT / "reports"
FRED_IDS = {"hy": "BAMLH0A0HYM2", "baa10y": "BAA10Y", "t10y2y": "T10Y2Y"}
YAHOO_IDS = {"spx": "^GSPC", "rut": "^RUT", "vix_daily": "^VIX", "vix3m_daily": "^VIX3M"}
START = "1990-01-01"


def _pct(x, d=0):
    return "" if pd.isna(x) else f"{x * 100:.{d}f}%"


def _num(x, d=0):
    return "" if pd.isna(x) else f"{x:.{d}f}"


def tbl(df: pd.DataFrame) -> str:
    return df.to_html(index=False, escape=False, border=0, classes="tbl")


def chart(sig: pd.DataFrame, spx: pd.Series, entries: dict[str, pd.DatetimeIndex]) -> str:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=spx.index, y=spx.values, name="S&P 500", line=dict(color="#58A6FF", width=1.2)), row=1, col=1)
    markers = {"capitulation": ("#2EA043", "triangle-up", "恐慌高峰"), "combo": ("#39D2C0", "diamond", "多重確認")}
    for name, dates in entries.items():
        color, symbol, label = markers[name]
        fig.add_trace(go.Scatter(x=dates, y=spx.reindex(dates).values, mode="markers", name=label,
                                 marker=dict(color=color, symbol=symbol, size=10, line=dict(color="#0D1117", width=1))), row=1, col=1)
    lvl = sig["warning_level"].where(sig["near_high"])
    fig.add_trace(go.Bar(x=lvl.index, y=lvl.values, name="早期警示（條件數）",
                         marker=dict(color=lvl.map(lambda v: "#F85149" if v >= 3 else ("#D29922" if v == 2 else "#30363D")).values),
                         marker_line_width=0), row=2, col=1)
    fig.update_yaxes(type="log", title_text="S&P 500（對數）", row=1, col=1)
    fig.update_yaxes(range=[0, 5], title_text="警示 0–5", row=2, col=1)
    fig.update_layout(height=640, bargap=0, template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
                      margin=dict(l=60, r=20, t=30, b=30), legend=dict(orientation="h", y=1.05))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    raw = load_many(FRED_IDS, YAHOO_IDS, CACHE_DIR, args.refresh)
    d = ds.build_daily_frame(raw)
    sig_full = ds.compute_signals(d)
    fwd_full = ds.forward_outcomes(d["spx"])
    sig, fwd, spx = sig_full.loc[START:], fwd_full.loc[START:], d["spx"].loc[START:]

    w_tbl = ds.warning_stats(sig, fwd)
    lead = ds.correction_lead_table(sig)
    lead20 = lead[lead["max_decline"] <= -0.20]
    e_sum, e_eps = ds.entry_stats(sig, fwd, spx)
    ep_stats = {lvl: ds.warning_episode_stats(sig, fwd, lvl) for lvl in (2, 3, 4)}

    pd.set_option("display.width", 200)
    print(w_tbl.round(3).to_string(index=False), "\n")
    print(ep_stats, "\n")
    print(lead.round(3).to_string(index=False), "\n")
    print(e_sum.round(3).to_string(index=False), "\n")
    for k, v in e_eps.items():
        print(k); print(v.round(3).to_string(index=False), "\n")
    latest = sig.iloc[-1]
    print("Latest", sig.index[-1].date(), latest.to_dict())

    # ---- report ----
    w_fmt = pd.DataFrame({
        "警示條件數": [("全部（基準）" if l == 0 else f"≥{l}") for l in w_tbl["level"]],
        "占接近高點時間": w_tbl["share_of_time"].map(_pct),
        "出現次數": w_tbl["episodes"].map(_num),
        "6 個月內跌 ≥10%": w_tbl["p_dd10"].map(_pct), "6 個月內跌 ≥20%": w_tbl["p_dd20"].map(_pct),
        "平均 6 個月報酬": w_tbl["avg_ret"].map(lambda x: _pct(x, 1)),
    })
    ep_fmt = pd.DataFrame([{
        "警示門檻": f"≥{l} 個條件", "出現次數": s["episodes"],
        "之後 6 個月跌 ≥10%": f"{s['hit10']} 次（{s['hit10'] / s['episodes']:.0%}）" if s["episodes"] else "",
        "之後 6 個月跌 ≥20%": f"{s['hit20']} 次（{s['hit20'] / s['episodes']:.0%}）" if s["episodes"] else "",
    } for l, s in ep_stats.items()])
    lead_fmt = pd.DataFrame({
        "跌破 -10% 的日期": lead["crossed_-10%"], "最終跌幅": lead["max_decline"].map(_pct),
        "≥2 條件提前（交易日）": lead["lead_days_L2"].map(_num), "≥3 條件提前（交易日）": lead["lead_days_L3"].map(_num),
    })
    names = {"baseline": "基準：任何跌 ≥15% 的日子", **{k: v.split("：")[0] for k, v in ds.ENTRY_SIGNALS.items()}}
    e_fmt = pd.DataFrame({
        "訊號": e_sum["signal"].map(names), "次數": e_sum["count"],
        "3 個月後上漲機率": e_sum["win_63"].map(_pct), "平均 3 個月報酬": e_sum["avg_63"].map(lambda x: _pct(x, 1)),
        "12 個月後上漲機率": e_sum["win_252"].map(_pct), "平均 12 個月報酬": e_sum["avg_252"].map(lambda x: _pct(x, 1)),
        "之後 6 個月平均再跌": e_sum["avg_further_drop_126"].map(lambda x: _pct(x, 1)),
        "最多再跌": e_sum["worst_further_drop_126"].map(lambda x: _pct(x, 1)),
    })
    ep_html = ""
    for k, v in e_eps.items():
        ep_html += f"<h3>{names[k]}：每一次訊號</h3>" + tbl(pd.DataFrame({
            "日期": v["date"], "當時跌幅": v["drawdown"].map(_pct),
            "相對真正低點": v["days_vs_trough"].map(lambda x: f"低點後 {x} 天" if x > 0 else ("就是低點" if x == 0 else f"低點前 {-x} 天")),
            "之後 6 個月再跌": v["further_drop_6m"].map(lambda x: _pct(x, 1)),
            "3 個月報酬": v["ret_3m"].map(lambda x: _pct(x, 1)), "12 個月報酬": v["ret_12m"].map(lambda x: _pct(x, 1)),
        }))
    cond_now = "".join(
        f"<li>{'🔴' if latest[f'w_{c}'] else '⚪'} {label}</li>" for c, label in ds.WARNING_CONDITIONS.items())
    entries = {k: ds.episode_starts(sig[f"e_{k}"]) for k in ds.ENTRY_SIGNALS}
    lvl = int(latest["warning_level"])
    html = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>早期警示與進場訊號</title>
<style>
body{{background:#0D1117;color:#E6EDF3;font-family:-apple-system,"Segoe UI","Noto Sans TC",sans-serif;max-width:1100px;margin:0 auto;padding:16px;line-height:1.6}}
h1{{font-size:24px}} h2{{font-size:19px;margin-top:40px;border-bottom:1px solid #30363D;padding-bottom:6px}} h3{{font-size:15px;margin-top:24px;color:#C9D1D9}}
.tbl{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums;display:block;overflow-x:auto}}
.tbl th,.tbl td{{padding:6px 10px;border-bottom:1px solid #21262D;text-align:right;white-space:nowrap}}
.tbl th:first-child,.tbl td:first-child{{text-align:left}} .tbl th{{color:#8B949E;font-weight:600}}
.note{{color:#8B949E;font-size:13px}} .now{{background:#161B22;border:1px solid #30363D;border-radius:8px;padding:12px 16px}}
.now b{{font-size:20px}} ul{{padding-left:20px}}
</style></head><body>
<h1>早期警示與進場訊號：歷史回測（{sig.index[0]:%Y-%m} – {sig.index[-1]:%Y-%m-%d}）</h1>
<div class="now">最新（{sig.index[-1]:%Y-%m-%d}）：早期警示 <b>{ds.WARNING_LEVEL_NAMES[lvl]} {lvl}/5</b>，
距一年高點 {latest['dd_1y']:+.1%}<ul>{cond_now}</ul></div>
<h2>走勢與訊號</h2>
<p class="note">下方長條：大盤距一年高點 10% 以內時的早期警示條件數（黃 = 注意，紅 = 警告）。上方標記：進場訊號。</p>
{chart(sig, spx, entries)}
<h2>早期警示：亮燈後真的會跌嗎？</h2>
<p class="note">只看大盤距一年高點 10% 以內的日子。「出現次數」把相隔不到 60 個交易日的訊號算同一次。</p>
{tbl(w_fmt)}
<h3>以「次」計算</h3>{tbl(ep_fmt)}
<h3>每次跌破 -10% 之前，有沒有提前亮燈（往前看 60 個交易日）</h3>
<p class="note">空白 = 沒有提前亮燈。</p>{tbl(lead_fmt)}
<h2>進場訊號：訊號後的表現</h2>
<p class="note">「之後 6 個月平均再跌」= 訊號當天之後，最低曾比訊號當天再低多少。基準是「只要跌超過 15% 就買」。</p>
{tbl(e_fmt)}{ep_html}
<h2>規則（事先決定，沒有針對歷史調參數）</h2>
<h3>早期警示條件</h3><ul class="note">{''.join(f'<li>{v}</li>' for v in ds.WARNING_CONDITIONS.values())}</ul>
<p class="note">0–1 個 = 正常，2 個 = 注意，3 個以上 = 警告。大盤已跌超過 10% 時不再發早期警示，改看進場訊號。</p>
<h3>進場訊號</h3><ul class="note">{''.join(f'<li>{v}</li>' for v in ds.ENTRY_SIGNALS.values())}</ul>
<h2>限制</h2><ul class="note">
<li>高收益債利差 2023 年前用 Baa 公司債利差換算；VIX3M 2006 年起才有，之前用 VIX 相對 50 日均值代替；小型股指數 1987 年 9 月起。</li>
<li>空頭次數少，所有機率都有很大誤差。歷史表現不保證未來。</li>
<li>這是統計提醒，不是投資建議。</li></ul>
</body></html>"""
    REPORT_DIR.mkdir(exist_ok=True)
    out = REPORT_DIR / "signals_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
