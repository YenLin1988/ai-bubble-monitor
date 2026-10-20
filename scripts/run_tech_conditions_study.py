"""Do tech-specific warning conditions improve Nasdaq-100 warnings? Pre-registered test.

Candidates (fixed before testing, mirroring the existing small-cap breadth rule):
    T1 semis        SOX lags the Nasdaq-100 by >= 5% over 60 trading days            (1994+)
    T2 tech breadth Nasdaq-100 equal-weight lags cap-weight by >= 5% over 60 days    (2006+)
    T3 tech fear    VXN / VIX in the top 10% of its trailing 1-year range            (2001+)

Adoption criteria (all must hold):
    1. more Nasdaq-100 -10% corrections preceded by a >=3 warning within 60 trading days
    2. share of >=3 warning episodes followed by a >=10% decline within 6 months not lower
    3. combo B Sharpe not lower in 1990-2007 or in 2008-2026

Usage:
    python scripts/run_tech_conditions_study.py [--refresh]
Writes reports/tech_conditions_study.html.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bubble_monitor import daily_signals as ds  # noqa: E402
from bubble_monitor import overlay as ov  # noqa: E402
from bubble_monitor.data_sources import load_many  # noqa: E402

CACHE_DIR = ROOT / "data" / "backtest_cache"
REPORT_DIR = ROOT / "reports"
FRED_IDS = {"hy": "BAMLH0A0HYM2", "baa10y": "BAA10Y", "t10y2y": "T10Y2Y", "tbill": "DTB3"}
YAHOO_IDS = {"spx": "^GSPC", "ndx": "^NDX", "rut": "^RUT", "vix_daily": "^VIX", "vix3m_daily": "^VIX3M",
             "sox": "^SOX", "ndxe": "^NDXE", "vxn_daily": "^VXN"}
START = "1990-01-01"
HALVES = {"1990–2007": ("1990-01-01", "2007-12-31"), "2008–2026": ("2008-01-01", "2026-12-31")}
CANDIDATES = {"T1": "半導體轉弱", "T2": "科技股廣度變窄", "T3": "科技股專屬恐慌"}


def _pct(x, d=0):
    return "" if pd.isna(x) else f"{x * 100:.{d}f}%"


def tbl(df: pd.DataFrame) -> str:
    return df.to_html(index=False, escape=False, border=0, classes="tbl")


def tech_conditions(d: pd.DataFrame, raw: dict) -> pd.DataFrame:
    idx = d.index
    ndx = raw["ndx"].reindex(idx).ffill(limit=2)

    def rel_drop(other):
        rel = other.reindex(idx).ffill(limit=2) / ndx
        chg = rel / rel.shift(60) - 1
        return (chg <= -0.05) & chg.notna()

    vxn = raw["vxn_daily"].reindex(idx).ffill(limit=2)
    ratio = vxn / d["vix"]
    t3 = (ratio >= ratio.rolling(252, min_periods=126).quantile(0.9)) & ratio.notna()
    return pd.DataFrame({"T1": rel_drop(raw["sox"]), "T2": rel_drop(raw["ndxe"]), "T3": t3}, index=idx)


def evaluate(sig: pd.DataFrame, fwd: pd.DataFrame, px: pd.Series, cash: pd.Series) -> dict:
    s, f = sig.loc[START:], fwd.loc[START:]
    lead = ds.correction_lead_table(s)
    st3 = ds.warning_episode_stats(s, f, 3)
    st4 = ds.warning_episode_stats(s, f, 4)
    entry = sig[["e_capitulation", "e_combo"]].fillna(False).astype(bool).any(axis=1)
    p = px.loc[START:].dropna()
    faber = ov.faber_exposure(px.dropna())
    warn4 = ov.warning_overlay(sig.dropna(subset=["dd_1y"]), 4)
    sim = ov.simulate(p, ov.combined_exposure(faber, warn4, entry), cash)
    out = {"caught": int(lead["lead_days_L3"].notna().sum()), "corrections": len(lead),
           "ep3": st3["episodes"], "hit3": st3["hit10"], "ep4": st4["episodes"], "hit4": st4["hit10"]}
    for name, (a, b) in HALVES.items():
        sub = sim.loc[a:b].copy()
        sub["equity"] = (1 + sub["ret"]).cumprod()
        m = ov.metrics(sub, cash)
        out[f"sharpe_{name}"] = m["sharpe"]
        out[f"mdd_{name}"] = m["max_dd"]
        out[f"cagr_{name}"] = m["cagr"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 220)

    raw = load_many(FRED_IDS, YAHOO_IDS, CACHE_DIR, args.refresh)
    d = ds.build_daily_frame(raw)
    cash = raw["tbill"]
    px = raw["ndx"].reindex(d.index).ffill(limit=2)
    base = ds.compute_signals(d, raw["ndx"])
    fwd = ds.forward_outcomes(px.ffill())
    tech = tech_conditions(d, raw)

    variants = {"現行 5 條件": []} | {f"+{k} {v}": [k] for k, v in CANDIDATES.items()} | {"+全部三個": list(CANDIDATES)}
    rows = {}
    for vname, extra in variants.items():
        sig = base.copy()
        if extra:
            sig["warning_level"] = sig["warning_level"] + tech[extra].sum(axis=1)
        rows[vname] = evaluate(sig, fwd, px, cash)
    res = pd.DataFrame(rows).T
    print(res.to_string(), "\n")
    for k in CANDIDATES:
        t = tech[k].loc[START:]
        print(k, "first active", t[t].index.min(), "share of days on", round(t.mean(), 3))

    cur = rows["現行 5 條件"]
    verdict_rows = []
    for vname, r in rows.items():
        if vname == "現行 5 條件":
            continue
        c1 = r["caught"] > cur["caught"]
        c2 = (r["hit3"] / max(r["ep3"], 1)) >= (cur["hit3"] / max(cur["ep3"], 1))
        c3 = all(r[f"sharpe_{h}"] >= cur[f"sharpe_{h}"] - 1e-9 for h in HALVES)
        verdict_rows.append({"版本": vname, "1. 多抓到修正": "✅" if c1 else "❌",
                             "2. 警告準確率不降": "✅" if c2 else "❌",
                             "3. 兩段夏普都不變差": "✅" if c3 else "❌",
                             "結論": "採用" if (c1 and c2 and c3) else "不採用"})
    verdict = pd.DataFrame(verdict_rows)
    print(verdict.to_string(index=False))

    show = pd.DataFrame({
        "版本": list(rows),
        "跌破 -10% 前有 ≥3 警告": [f"{r['caught']}/{r['corrections']}" for r in rows.values()],
        "≥3 警告 → 6M 內跌 ≥10%": [f"{r['hit3']}/{r['ep3']}（{r['hit3'] / max(r['ep3'], 1):.0%}）" for r in rows.values()],
        "≥4 警告 → 6M 內跌 ≥10%": [f"{r['hit4']}/{r['ep4']}（{r['hit4'] / max(r['ep4'], 1):.0%}）" for r in rows.values()],
        **{f"組合 B 夏普 {h}": [f"{r[f'sharpe_{h}']:.2f}" for r in rows.values()] for h in HALVES},
        **{f"組合 B 最大回撤 {h}": [_pct(r[f'mdd_{h}']) for r in rows.values()] for h in HALVES},
        **{f"組合 B 年化 {h}": [_pct(r[f'cagr_{h}'], 1) for r in rows.values()] for h in HALVES},
    })
    html = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>科技股專屬條件</title>
<style>
body{{background:#0D1117;color:#E6EDF3;font-family:-apple-system,"Segoe UI","Noto Sans TC",sans-serif;max-width:1100px;margin:0 auto;padding:16px;line-height:1.6}}
h1{{font-size:24px}} h2{{font-size:19px;margin-top:36px;border-bottom:1px solid #30363D;padding-bottom:6px}}
.tbl{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums;display:block;overflow-x:auto}}
.tbl th,.tbl td{{padding:6px 10px;border-bottom:1px solid #21262D;text-align:right;white-space:nowrap}}
.tbl th:first-child,.tbl td:first-child{{text-align:left}} .tbl th{{color:#8B949E;font-weight:600}}
.note{{color:#8B949E;font-size:13px}}
</style></head><body>
<h1>科技股專屬警示條件：那斯達克 100 測試（1990–2026）</h1>
<p class="note">候選條件與採用標準都在測試前固定。T1 半導體（SOX）60 日落後那斯達克 100 ≥5%（1994 起）；
T2 那斯達克 100 等權重 60 日落後市值加權 ≥5%（2006 起）；T3 VXN/VIX 比值在過去一年前 10%（2001 起）。</p>
<h2>結論</h2>{tbl(verdict)}
<h2>詳細數字</h2>{tbl(show)}
<p class="note">採用標準：1. 跌破 -10% 前 60 個交易日內有 ≥3 警告的次數增加；2. ≥3 警告後 6 個月內跌 ≥10% 的比例不降；
3. 組合 B（均線 × 警示 ≥4 + 恐慌進場）在 1990–2007、2008–2026 兩段的夏普值都不變差。歷史回測不保證未來，這不是投資建議。</p>
</body></html>"""
    REPORT_DIR.mkdir(exist_ok=True)
    out = REPORT_DIR / "tech_conditions_study.html"
    out.write_text(html, encoding="utf-8")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
