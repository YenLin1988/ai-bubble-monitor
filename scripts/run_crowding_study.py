"""Test whether investor crowding predicts weaker S&P 500 returns / bigger drawdowns.

Usage:
    python scripts/run_crowding_study.py [--refresh]
Writes reports/crowding_study.html.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bubble_monitor import crowding as cr  # noqa: E402
from bubble_monitor.backtest import find_drawdown_events  # noqa: E402
from bubble_monitor.data_sources import cached, load_finra_margin, load_many  # noqa: E402

CACHE_DIR = ROOT / "data" / "backtest_cache"
REPORT_DIR = ROOT / "reports"
FRED_IDS = {"household_alloc": "BOGZ1FL153064486Q", "corp_equity": "NCBEILQ027S",
            "total_debt": "TCMDO", "mmf": "MMMFFAQ027S"}
HORIZONS = (3, 12, 36)
SPLIT = "1990-01-01"


def _pct(x, d=0):
    return "" if pd.isna(x) else f"{x * 100:.{d}f}%"


def _corr(c, n):
    return f"{c:+.2f}（{n}）" if not pd.isna(c) else "資料不足"


def tbl(df: pd.DataFrame) -> str:
    return df.to_html(index=False, escape=False, border=0, classes="tbl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    raw = load_many(FRED_IDS, {"spx": "^GSPC"}, CACHE_DIR, args.refresh)
    finra = None

    def finra_col(col):
        nonlocal finra
        if finra is None:
            finra = load_finra_margin()
        return finra[col]

    raw["margin_debt"] = cached(CACHE_DIR, "finra_margin_debt", lambda: finra_col("margin_debt"), args.refresh)
    raw["free_credit"] = cached(CACHE_DIR, "finra_free_credit", lambda: finra_col("free_credit"), args.refresh)

    spx = raw["spx"]
    index = pd.date_range("1950-01-31", spx.index[-1], freq="ME")
    spx_m = cr.asof_monthly(spx, index)
    meas = cr.build_measures(raw, index)
    pct = meas.apply(cr.expanding_percentile)
    fwd = cr.forward_monthly(spx_m, HORIZONS)
    bears = [e for e in find_drawdown_events(spx, 0.20) if e.peak_date.year >= 1955]
    early, late = meas.index < SPLIT, meas.index >= SPLIT

    corr_rows, quint_html, peak_rows = [], "", []
    for key, (label, desc) in cr.MEASURES.items():
        v = meas[key].dropna()
        if not len(v):
            continue
        row = {"指標": label, "資料起始": v.index[0].strftime("%Y-%m"), "最新": f"{v.iloc[-1]:.1f}",
               "最新歷史百分位": _pct(pct[key].dropna().iloc[-1])}
        for h in HORIZONS:
            row[f"與 {h}M 報酬相關（n）"] = _corr(*cr.rank_corr(meas[key], fwd[f"ret_{h}"], step=h))
        c, n = cr.rank_corr(meas[key], fwd["maxdd_12"], step=12)
        row["與 12M 回撤深度相關"] = f"{-c:+.2f}" if not pd.isna(c) else ""
        for tag, mask in (("1990 前", early), ("1990 後", late)):
            row[f"12M 相關：{tag}"] = _corr(*cr.rank_corr(meas[key][mask], fwd["ret_12"][mask], step=12))
        row["36M 相關：1990 後"] = _corr(*cr.rank_corr(meas[key][late], fwd["ret_36"][late], step=36))
        corr_rows.append(row)

        q = cr.quintile_table(pct[key], fwd, 12)
        quint_html += f"<h3>{label}</h3><p class='note'>{desc}</p>" + tbl(pd.DataFrame({
            "擁擠程度（跟自己過去比）": q.index.astype(str), "月數": q["months"].values,
            "之後 12M 平均報酬": q["avg_ret"].map(lambda x: _pct(x, 1)).values,
            "12M 後上漲機率": q["p_up"].map(_pct).values,
            "12M 內平均最大回撤": q["avg_maxdd"].map(lambda x: _pct(x, 1)).values,
            "12M 內跌 ≥20% 機率": q["p_dd20"].map(_pct).values,
        }))
        print(label)
        print(q.round(3).to_string(), "\n")

    for ev in bears:
        pm = ev.peak_date + pd.offsets.MonthEnd(0)
        r = {"空頭高點": ev.peak_date.date(), "跌幅": _pct(ev.depth)}
        for key, (label, _) in cr.MEASURES.items():
            r[label] = _pct(pct[key].get(pm, np.nan))
        peak_rows.append(r)

    corr = pd.DataFrame(corr_rows)
    peaks = pd.DataFrame(peak_rows)
    pd.set_option("display.width", 250)
    print(corr.to_string(index=False), "\n")
    print(peaks.to_string(index=False))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=meas.index, y=meas["household_alloc"], name="家庭股票配置 %",
                             line=dict(color="#F0883E", width=1.8)), secondary_y=False)
    fig.add_trace(go.Scatter(x=fwd.index, y=fwd["ret_36"] * 100, name="之後 3 年 S&P 500 報酬 %",
                             line=dict(color="#58A6FF", width=1.2)), secondary_y=True)
    for ev in bears:
        fig.add_vrect(x0=ev.peak_date, x1=ev.trough_date, fillcolor="rgba(248,81,73,0.15)", line_width=0)
    fig.update_yaxes(title_text="家庭股票配置 %", secondary_y=False)
    fig.update_yaxes(title_text="之後 3 年報酬 %", secondary_y=True)
    fig.update_layout(height=480, template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
                      margin=dict(l=60, r=60, t=30, b=30), legend=dict(orientation="h", y=1.1))

    latest = meas.dropna(how="all").index[-1]
    html = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>散戶擁擠度研究</title>
<style>
body{{background:#0D1117;color:#E6EDF3;font-family:-apple-system,"Segoe UI","Noto Sans TC",sans-serif;max-width:1100px;margin:0 auto;padding:16px;line-height:1.6}}
h1{{font-size:24px}} h2{{font-size:19px;margin-top:36px;border-bottom:1px solid #30363D;padding-bottom:6px}} h3{{font-size:15px;margin-top:22px}}
.tbl{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums;display:block;overflow-x:auto}}
.tbl th,.tbl td{{padding:6px 10px;border-bottom:1px solid #21262D;text-align:right;white-space:nowrap}}
.tbl th:first-child,.tbl td:first-child{{text-align:left}} .tbl th{{color:#8B949E;font-weight:600}}
.note{{color:#8B949E;font-size:13px}}
</style></head><body>
<h1>散戶擁擠度與之後大盤表現（資料至 {latest:%Y-%m}）</h1>
<p class="note">假設：散戶越擁擠，之後報酬越差、越容易大跌。每個指標都對齊實際公布時間，擁擠程度只跟自己的過去比（不偷看未來）。紅色區塊為跌 ≥20% 的空頭。</p>
<h2>家庭股票配置 vs 之後 3 年報酬</h2>{fig.to_html(full_html=False, include_plotlyjs="cdn")}
<h2>相關性總表</h2>
<p class="note">Spearman 等級相關，只用不重疊的樣本（例如 12M 每 12 個月取一筆），括號內是樣本數。負值 = 越擁擠、之後報酬越差（符合假設）。「與 12M 回撤深度相關」正值 = 越擁擠、之後回撤越深。樣本數少於約 20 時，±0.4 以內都可能只是雜訊。</p>
{tbl(corr)}
<h2>依擁擠程度分五組：之後 12 個月</h2>{quint_html}
<h2>每次空頭開始時的擁擠程度（歷史百分位）</h2>{tbl(peaks)}
<h2>限制</h2><ul class="note">
<li>AAII 散戶情緒調查、CBOE Put/Call 比率的網站無法自動下載，沒有納入；若你手動下載 AAII 的 sentiment.xls，可以加進來。</li>
<li>家庭配置、AIAE 是季資料，3 年期的不重疊樣本只有約 20 筆；融資資料 1997 年才開始。</li>
<li>報酬為 S&amp;P 500 價格報酬，不含股息。這是歷史統計，不是投資建議。</li></ul>
</body></html>"""
    REPORT_DIR.mkdir(exist_ok=True)
    out = REPORT_DIR / "crowding_study.html"
    out.write_text(html, encoding="utf-8")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
