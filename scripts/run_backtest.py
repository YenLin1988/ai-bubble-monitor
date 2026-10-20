"""Backtest the v1 and v2 stress models against S&P 500 drawdowns and write an HTML report.

Usage:
    python scripts/run_backtest.py            # uses cached data if < 1 day old
    python scripts/run_backtest.py --refresh  # force re-download
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

from bubble_monitor import backtest as bt  # noqa: E402
from bubble_monitor.data_sources import load_many  # noqa: E402
from bubble_monitor.model_config import RISK_LABELS_ZH  # noqa: E402
from bubble_monitor.model_v2 import V2_FACTORS, V2_LABELS_ZH, V2_WEIGHTS, score_factors_v2  # noqa: E402

CACHE_DIR = ROOT / "data" / "backtest_cache"
REPORT_DIR = ROOT / "reports"

FRED_IDS = {
    "gs10": "GS10", "fedfunds": "FEDFUNDS", "walcl": "WALCL", "wtregen": "WTREGEN",
    "rrp": "RRPONTSYD", "m2": "M2SL", "usdjpy": "DEXJPUS", "t10y2y": "T10Y2Y",
    "hy": "BAMLH0A0HYM2", "baa10y": "BAA10Y", "baa": "BAA", "nfci": "NFCI", "corp_equity": "NCBEILQ027S",
    "gdp": "GDP", "vxn": "VXNCLS", "vix": "VIXCLS", "vix3m": "VXVCLS", "umcsent": "UMCSENT",
    "oil": "DCOILWTICO", "oil_monthly": "WTISPLC", "dollar_broad_old": "DTWEXM", "unrate": "UNRATE",
}
YAHOO_IDS = {"spx": "^GSPC", "dxy": "DX-Y.NYB", "gold": "GC=F", "rut": "^RUT"}

HISTORY_START = "1960-01-31"  # warm-up for v2 trailing percentiles
START = "1986-01-31"          # first scored month
MIN_COVERAGE = 0.70
NEAR_HIGH = 0.95  # "market still near its high" = within 5% of the all-time high
CRASH_DEPTHS = (0.20, 0.15)
PERIODS = {"1986_2005": ("1986-01-01", "2005-12-31"), "2006_2026": ("2006-01-01", "2026-12-31")}
BUCKETS = (0, 20, 30, 40, 50, 60, 101)
COLORS = {"v1": "#8B949E", "v2": "#F0883E"}


def load_raw(refresh: bool) -> dict[str, pd.Series]:
    return load_many(FRED_IDS, YAHOO_IDS, CACHE_DIR, refresh)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _pct(x, digits=0):
    return "" if pd.isna(x) else f"{x * 100:.{digits}f}%"


def _num(x, digits=0):
    return "" if pd.isna(x) else f"{x:.{digits}f}"


def html_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, escape=False, border=0, classes="tbl")


def fmt_events(tbl: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "高點": tbl["peak"], "低點": tbl["trough"], "跌幅": tbl["depth"].map(_pct),
        "12 個月前": tbl["score_12m_before"].map(_num), "6 個月前": tbl["score_6m_before"].map(_num),
        "高點當月": tbl["score_at_peak"].map(_num), "前 12 個月最高": tbl["max_score_12m_before"].map(_num),
    })


def fmt_thresholds(tbl: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "門檻": tbl["threshold"], "警報時間占比": tbl["time_in_alert"].map(_pct),
        "精準度": tbl["precision"].map(_pct), "倍數": tbl["lift"].map(lambda x: _num(x, 2)),
        "精準度（接近高點時）": tbl["precision_near_high"].map(_pct),
        "倍數（接近高點時）": tbl["lift_near_high"].map(lambda x: _num(x, 2)),
        "事件捕捉": tbl["events_caught"],
    })


def fmt_buckets(tbl: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "分數區間": [f"{int(i.left)}–{min(int(i.right), 100)}" for i in tbl.index], "月數": tbl["months"].values,
        "平均 12M 報酬": tbl["avg_fwd_12m_return"].map(lambda x: _pct(x, 1)).values,
        "平均 12M 最大回撤": tbl["avg_fwd_12m_max_dd"].map(lambda x: _pct(x, 1)).values,
        "跌 ≥10% 機率": tbl["p_dd_10"].map(_pct).values, "跌 ≥20% 機率": tbl["p_dd_20"].map(_pct).values,
    })


def fmt_factors(tbl: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({
        "因子": [f"{labels.get(f, f)} {f}" for f in tbl["factor"]], "資料起始": tbl["first_month"],
        "卡在 100": tbl["pct_at_100"].map(_pct), "卡在 0": tbl["pct_at_0"].map(_pct),
        "空頭前 12M 平均": tbl["avg_pre_crash"].map(_num), "其他時間平均": tbl["avg_other"].map(_num),
        "與未來回撤相關": tbl["corr_with_fwd_dd"].map(lambda x: _num(x, 2)),
    })


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _dark(fig: go.Figure, **kw) -> go.Figure:
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#0D1117", **kw)
    return fig


def build_chart(spx_m: pd.Series, stresses: dict[str, pd.Series], events) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=spx_m.index, y=spx_m.values, name="S&P 500", line=dict(color="#58A6FF", width=1.5)), row=1, col=1)
    for name, s in stresses.items():
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=f"{name} 壓力分數",
                                 line=dict(color=COLORS[name], width=1.6 if name == "v2" else 1.1)), row=2, col=1)
    for ev in events:
        for r in (1, 2):
            fig.add_vrect(x0=ev.peak_date, x1=ev.trough_date, fillcolor="rgba(248,81,73,0.15)", line_width=0, row=r, col=1)
    fig.update_yaxes(type="log", title_text="S&P 500（對數）", row=1, col=1)
    fig.update_yaxes(range=[0, 100], title_text="壓力 0–100", row=2, col=1)
    return _dark(fig, height=640, margin=dict(l=60, r=20, t=30, b=30), legend=dict(orientation="h", y=1.06))


def build_heatmap(scores: pd.DataFrame, labels: dict[str, str], events) -> go.Figure:
    z = scores.T
    fig = go.Figure(go.Heatmap(z=z.values, x=z.columns, y=[f"{labels.get(n, n)}" for n in z.index],
                               colorscale="RdYlGn_r", zmin=0, zmax=100, colorbar=dict(title="分數")))
    for ev in events:
        fig.add_vline(x=ev.peak_date, line=dict(color="white", width=1, dash="dot"))
    return _dark(fig, height=40 + 30 * len(z), margin=dict(l=140, r=20, t=20, b=30))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(stress, scores, spx, near_high, fwd_dd, fwd_ret, crash) -> dict:
    events = [e for e in bt.find_drawdown_events(spx, crash) if e.peak_date >= stress.index[0] - pd.DateOffset(months=1)]
    return {
        "events": events,
        "events_tbl": bt.event_table(stress, events),
        "thresh_tbl": bt.threshold_table(stress, fwd_dd, events, thresholds=range(10, 85, 5), crash_depth=crash, near_high=near_high),
        "bucket_tbl": bt.bucket_table(stress, fwd_ret, fwd_dd, bins=BUCKETS),
        "factor_tbl": bt.factor_diagnostics(scores, fwd_dd, events),
        "summary": bt.model_summary(stress, fwd_dd, events, near_high, crash, PERIODS),
    }


def comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    spec = [
        ("AUC：全部月份", "auc_all", lambda x: _num(x, 2)),
        ("AUC：接近高點時", "auc_near_high", lambda x: _num(x, 2)),
        ("AUC：接近高點時，1986–2005", "auc_near_high_1986_2005", lambda x: _num(x, 2)),
        ("AUC：接近高點時，2006–2026", "auc_near_high_2006_2026", lambda x: _num(x, 2)),
        ("警報線（各模型自己的第 80 百分位）", "alert_line", lambda x: _num(x, 1)),
        ("接近高點時亮燈 → 12M 內真的崩", "precision_near_high_at_line", _pct),
        ("崩盤前 12 個月有亮燈", "events_caught_at_line", str),
    ]
    for crash in CRASH_DEPTHS:
        for label, key, f in spec:
            rows.append({
                "崩盤定義": f"跌 ≥{int(crash * 100)}%", "指標": label,
                "v1（現行）": f(results[("v1", crash)]["summary"][key]),
                "v2": f(results[("v2", crash)]["summary"][key]),
            })
    return pd.DataFrame(rows)


def model_section(title: str, res: dict, labels: dict[str, str], heatmap: str, crash_pct: int) -> str:
    return f"""
<h3>每次空頭前的分數</h3>{html_table(fmt_events(res['events_tbl']))}
<h3>門檻訊號品質（崩盤 = 跌 ≥{crash_pct}%）</h3>{html_table(fmt_thresholds(res['thresh_tbl']))}
<h3>分數區間與之後 12 個月表現</h3>{html_table(fmt_buckets(res['bucket_tbl']))}
<h3>個別因子診斷</h3>{html_table(fmt_factors(res['factor_tbl'], labels))}
<h3>因子分數熱圖</h3>{heatmap}"""


def write_report(path: Path, ctx: dict) -> None:
    r2, r1 = ctx["results"][("v2", 0.20)], ctx["results"][("v1", 0.20)]
    v2_defs = pd.DataFrame([
        {"因子": f"{zh} {name}", "群組": cluster, "權重": _pct(V2_WEIGHTS[name], 1), "定義": desc}
        for name, (zh, cluster, _, desc, _) in V2_FACTORS.items()
    ])
    html = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>壓力模型回測</title>
<style>
body{{background:#0D1117;color:#E6EDF3;font-family:-apple-system,"Segoe UI","Noto Sans TC",sans-serif;max-width:1100px;margin:0 auto;padding:16px;line-height:1.6}}
h1{{font-size:24px}} h2{{font-size:19px;margin-top:40px;border-bottom:1px solid #30363D;padding-bottom:6px}} h3{{font-size:15px;margin-top:26px;color:#C9D1D9}}
.tbl{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums;display:block;overflow-x:auto}}
.tbl th,.tbl td{{padding:6px 10px;border-bottom:1px solid #21262D;text-align:right;white-space:nowrap}}
.tbl th:first-child,.tbl td:first-child,.tbl td:nth-child(2){{text-align:left}} .tbl th{{color:#8B949E;font-weight:600}}
.note{{color:#8B949E;font-size:13px}} .kpi{{display:flex;gap:12px;flex-wrap:wrap}}
.kpi div{{background:#161B22;border:1px solid #30363D;border-radius:8px;padding:10px 14px}} .kpi b{{display:block;font-size:20px}}
details{{margin-top:20px}} summary{{cursor:pointer;font-size:17px;font-weight:600}}
</style></head><body>
<h1>壓力模型回測：v1（現行）vs v2（{ctx['start']} – {ctx['end']}）</h1>
<p class="note">逐月重建兩個模型的歷史分數，對照 S&amp;P 500 的空頭市場。紅色區塊為高點到低點期間。</p>
<div class="kpi">
<div>回測月數<b>{ctx['n_months']}</b></div>
<div>跌 ≥20% 空頭<b>{len(r2['events'])} 次</b></div>
<div>v1 最新分數<b>{_num(ctx['latest']['v1'], 1)}</b></div>
<div>v2 最新分數<b>{_num(ctx['latest']['v2'], 1)}</b></div>
<div>v2 警報線<b>{_num(r2['summary']['alert_line'], 1)}</b></div>
</div>
<h2>模型比較</h2>
<p class="note">AUC = 隨機挑一個「之後 12 個月會崩」的月份，它的分數比隨機一個「不會崩」的月份高的機率。0.5 = 擲硬幣，越高越好，不需要選門檻。
「接近高點時」只看 S&amp;P 500 距歷史高點 5% 以內的月份，也就是真正「崩盤前」的情境，排除崩盤進行中才亮燈。
警報線取各模型自己分數的第 80 百分位，讓兩個模型都有 20% 的時間在警報，公平比較。</p>
{html_table(ctx['comparison'])}
<h2>S&amp;P 500 與兩個模型的壓力分數</h2>{ctx['chart']}
<h2>v2 因子定義</h2>
<p class="note">規則依經濟邏輯事先決定，沒有針對這 5 次崩盤調參數。百分位只取過去資料，中位數→0 分、第 95 百分位→100 分。</p>
{html_table(v2_defs)}
<h2>v2 詳細結果</h2>{model_section('v2', r2, V2_LABELS_ZH, ctx['heat_v2'], 20)}
<details><summary>v1（現行模型）詳細結果</summary>{model_section('v1', r1, RISK_LABELS_ZH, ctx['heat_v1'], 20)}</details>
<h2>方法與限制</h2><ul class="note">{''.join(f'<li>{n}</li>' for n in ctx['notes'])}</ul>
</body></html>"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    raw = load_raw(args.refresh)
    spx = raw["spx"]
    full_index = pd.date_range(HISTORY_START, spx.index[-1], freq="ME")  # complete months only
    values = bt.build_factor_values(raw, full_index)

    scores_v1 = bt.score_factors(values)
    stress_v1, cov_v1 = bt.composite_stress(scores_v1)
    scores_v2 = score_factors_v2(values)
    stress_v2, cov_v2 = bt.composite_stress(scores_v2, V2_WEIGHTS)

    valid = (full_index >= START) & (cov_v1 >= MIN_COVERAGE) & (cov_v2 >= MIN_COVERAGE)
    idx = full_index[valid]
    models = {"v1": (stress_v1[idx], scores_v1.loc[idx]), "v2": (stress_v2[idx], scores_v2.loc[idx])}

    spx_m = bt.month_end_last(spx, idx)
    near_high = spx_m >= NEAR_HIGH * bt.month_end_last(spx.cummax(), idx)
    fwd_dd = bt.forward_max_drawdown(spx, idx, 12)
    fwd_ret = bt.forward_return(spx, idx, 12)

    results = {}
    for name, (stress, scores) in models.items():
        for crash in CRASH_DEPTHS:
            results[(name, crash)] = evaluate(stress, scores, spx, near_high, fwd_dd, fwd_ret, crash)

    REPORT_DIR.mkdir(exist_ok=True)
    pd.concat(
        [stress_v1[idx].rename("stress_v1"), stress_v2[idx].rename("stress_v2"),
         scores_v1.loc[idx].add_prefix("v1_"), scores_v2.loc[idx].add_prefix("v2_"), values.loc[idx]], axis=1,
    ).to_csv(REPORT_DIR / "backtest_monthly.csv", float_format="%.4f")

    # ---- console summary ----
    pd.set_option("display.width", 220)
    comp = comparison_table(results)
    print(f"Months: {len(idx)} ({idx[0].date()} .. {idx[-1].date()})  "
          f"latest v1={stress_v1[idx].iloc[-1]:.1f} v2={stress_v2[idx].iloc[-1]:.1f}\n")
    print(comp.to_string(index=False), "\n")
    for name in models:
        r = results[(name, 0.20)]
        print(f"== {name} (crash >= 20%) ==")
        print(r["events_tbl"].round(1).to_string(index=False))
        print(r["thresh_tbl"].round(3).to_string(index=False))
        print(r["factor_tbl"].round(2).to_string(index=False), "\n")

    # ---- report ----
    events20 = results[("v2", 0.20)]["events"]
    notes = [
        f"資料：FRED 與 Yahoo Finance，月底取樣；兩個模型的可用權重都 ≥ {MIN_COVERAGE:.0%} 的月份才納入比較，缺的因子當月權重重新分配。",
        "v1：AI 估值（EV/EBITDA）沒有歷史資料，整段排除。淨流動性 2002 年底才有資料；VIX 期限結構 2008 年起、市場廣度（Russell 2000）1988 年起才有。",
        f"高收益債利差 FRED 只提供近 3 年，之前用 Baa 公司債利差換算（HY ≈ {bt.HY_FROM_BAA_SLOPE}×Baa {bt.HY_FROM_BAA_INTERCEPT:+}），1986 年前用月資料 Baa − GS10。",
        "巴菲特指標用 Fed Z.1 企業股權市值 ÷ GDP，以 S&P 500 外推到當月（現行 App 用的 WILL5000PR 已從 FRED 下架）。",
        "VXN 2001 年前用 VIX；美元 1985 年前用 Fed 舊版貿易加權美元（DTWEXM）的變動率。",
        "已處理公布時間落差（M2、失業率延後 1 個月；GDP、Z.1 依季度公布時間），但 FRED 提供的是修正後資料，仍有修正偏誤（NFCI、GDP 最明顯）。",
        "v2 的規則是看過 v1 回測結果後才設計的；雖然沒有針對崩盤日期調參數，仍不算真正的樣本外驗證。分兩段期間的 AUC 是穩定度的參考。",
        "跌 ≥20% 的空頭只有少數幾次，跌 ≥15% 多幾次可當穩健性檢查；所有統計數字的不確定性都很大。",
    ]
    report_path = REPORT_DIR / "backtest_report.html"
    write_report(report_path, {
        "start": idx[0].strftime("%Y-%m"), "end": idx[-1].strftime("%Y-%m"), "n_months": len(idx),
        "latest": {n: s.iloc[-1] for n, (s, _) in models.items()},
        "results": results, "comparison": comp,
        "chart": build_chart(spx_m, {n: s for n, (s, _) in models.items()}, events20).to_html(full_html=False, include_plotlyjs="cdn"),
        "heat_v1": build_heatmap(models["v1"][1], RISK_LABELS_ZH, events20).to_html(full_html=False, include_plotlyjs=False),
        "heat_v2": build_heatmap(models["v2"][1], V2_LABELS_ZH, events20).to_html(full_html=False, include_plotlyjs=False),
        "notes": notes,
    })
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
