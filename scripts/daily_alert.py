"""Daily early-warning / entry-signal check. Run once per trading day after the US close.

Usage:
    python scripts/daily_alert.py            # print status, send alerts only when a signal changes
    python scripts/daily_alert.py --status   # also send today's status even if nothing changed
    python scripts/daily_alert.py --weekly   # also send the weekly summary (automatic on scheduled Friday runs)
    python scripts/daily_alert.py --dry-run  # print what would be sent, send nothing, do not log

Channels come from environment variables (or a local .env file); see .env.example.
Sending requires ALERTS_ENABLED=1. ALERT_INDEX picks the tracked index: sox (default), ndx or spx.
Every non-dry run appends today's signals to data/signal_log.csv (the forward-test record).
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bubble_monitor import daily_signals as ds  # noqa: E402
from bubble_monitor import signal_log  # noqa: E402
from bubble_monitor.alerts import Alert  # noqa: E402
from bubble_monitor.health import load_with_health  # noqa: E402
from bubble_monitor.notifications import dispatch_alerts, enabled_channels  # noqa: E402

CACHE_DIR = ROOT / "data" / "backtest_cache"
LOG_PATH = ROOT / "data" / "signal_log.csv"
FRED_IDS = {"hy": "BAMLH0A0HYM2", "baa10y": "BAA10Y", "t10y2y": "T10Y2Y"}
YAHOO_IDS = {"spx": "^GSPC", "ndx": "^NDX", "sox": "^SOX", "rut": "^RUT", "vix_daily": "^VIX", "vix3m_daily": "^VIX3M"}
ESSENTIAL = ("spx", "baa10y")  # without these no signal can be computed
SEVERITY = {"warning_up": "warning", "warning_clear": "info", "entry": "info", "trend": "info",
            "exit": "critical", "exit_watch": "warning",
            "position": "warning", "data": "warning"}

# Backtest evidence quoted in position-change alerts (scripts/run_index_study.py, price index, 1990-2026).
INDEXES = {
    "sox": ("費城半導體 SOX", "費城半導體 1994–2026：年化 19.5% 對一直持有 15.1%，最大回撤 -51% 對 -87%；"
                           "2008 年後報酬相近（19.6% 對 19.3%）、最大回撤 -33% 對 -59%。"
                           "保護主要來自 10 個月均線，早期警示對 SOX 幾乎沒有鑑別力"),
    "ndx": ("那斯達克 100","那斯達克 100 1990–2026：年化報酬與一直持有相同（14.1%），最大回撤 -39% 對 -83%；"
                         "但 2008 年後每年約少賺 2 個百分點"),
    "spx": ("S&P 500", "S&P 500 1990–2026：年化報酬與一直持有相近，最大回撤 -21% 對 -55%；8 個海外市場夏普值全部變好"),
}


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def status_text(name: str, sig, pos, others: list[tuple[str, object]] = ()) -> str:
    today = sig.iloc[-1]
    lvl = int(today["warning_level"])
    lines = [f"{sig.index[-1]:%Y-%m-%d} {name} 狀態",
             f"距一年高點 {today['dd_1y']:+.1%}，VIX {today['vix']:.1f}"]
    if today["near_high"]:
        lines.append(f"早期警示：{ds.WARNING_LEVEL_NAMES[lvl]}（{lvl}/5）")
        lines += [f"  {'●' if today[f'w_{c}'] else '○'} {label}" for c, label in ds.WARNING_CONDITIONS.items()]
    else:
        lines.append("已進入修正區（跌幅 >10%），改看進場訊號：")
        lines += [f"  {'●' if today[f'e_{k}'] else '○'} {label}" for k, label in ds.ENTRY_SIGNALS.items()]
    ma10, price = pos.attrs["ma10_provisional"], pos.attrs["price"]
    gap = price / ma10 - 1
    if pos["trend"].iloc[-1] >= 1:
        lines.append(f"出場訊號（10 個月均線）：未觸發；目前價格距均線 {gap:+.1%}，"
                     "月底收盤跌破均線才會出場")
    else:
        lines.append(f"出場訊號（10 個月均線）：已觸發，空手中；目前價格距均線 {gap:+.1%}，"
                     "月底收盤站回均線才會重新進場")
    lines.append(f"規則部位（均線 × 警示 + 恐慌進場）：{pos['target'].iloc[-1]:.0%}")
    for other_name, other_sig in others:
        o = other_sig.iloc[-1]
        lines.append(f"參考 {other_name}：距一年高點 {o['dd_1y']:+.1%}，早期警示 {int(o['warning_level'])}/5")
    return "\n".join(lines)


def events_for(sig, fwd, target, evidence: str, min_level: int):
    """Events for the last row of ``sig``."""
    events = ds.todays_events(sig, fwd, min_warning_level=min_level)
    t = target.loc[: sig.index[-1]].dropna()
    pos = ds.position_frame(t, sig.loc[t.index])
    events += ds.position_events(pos, f"{sig.index[-1]:%Y-%m-%d}", evidence)
    return events, pos


def weekly_text(name, sig, fwd, target, evidence, min_level, status, problems, card_text) -> str:
    lines = ["本週摘要（每週五寄出；沒收到這封信代表系統可能停擺，請到 GitHub Actions 檢查）", ""]
    week = []
    for i in range(max(len(sig) - 5, 2), len(sig) + 1):
        evs, _ = events_for(sig.iloc[:i], fwd.iloc[:i], target, evidence, min_level)
        week += [f"  {sig.index[i - 1]:%m/%d} {e.title}" for e in evs]
    lines.append("本週訊號變化：" if week else "本週訊號變化：無")
    lines += week
    lines += ["", status, "", card_text, "",
              "資料狀態：" + ("全部正常" if not problems else "；".join(problems))]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="send today's status even without a new signal")
    ap.add_argument("--weekly", action="store_true", help="send the weekly summary")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")

    # GitHub Actions passes unset variables as empty strings, so fall back on "" too.
    index_key = (os.environ.get("ALERT_INDEX") or "sox").lower().lstrip("^")
    if index_key not in INDEXES:
        sys.exit(f"ALERT_INDEX must be one of {list(INDEXES)}, got {index_key!r}")
    name, evidence = INDEXES[index_key]
    min_level = int(os.environ.get("ALERT_MIN_WARNING_LEVEL") or "3")
    now = dt.datetime.now(dt.timezone.utc)
    weekly = args.weekly or (os.environ.get("GITHUB_EVENT_NAME") == "schedule" and now.weekday() == 4)

    raw, problems = load_with_health(FRED_IDS, YAHOO_IDS, CACHE_DIR, 6 * 3600, now.date())
    for p in problems:
        print("資料問題：", p)
    missing = [k for k in (*ESSENTIAL, index_key) if k not in raw]
    if missing:
        alert = Alert(f"data-fatal-{now:%Y-%m-%d}", "critical", "資料異常：今天無法計算訊號",
                      "缺少必要資料：" + "、".join(missing) + "。\n" + "\n".join(problems))
        print(f"[CRITICAL] {alert.title}\n{alert.message}")
        if not args.dry_run:
            dispatch_alerts([alert])
        sys.exit(1)

    d = ds.build_daily_frame(raw)
    target = raw[index_key].reindex(d.index).ffill(limit=2)
    sig = ds.compute_signals(d, None if index_key == "spx" else raw[index_key])
    fwd = ds.forward_outcomes(target.ffill())
    events, pos = events_for(sig, fwd, target, evidence, min_level)

    others = [(INDEXES[k][0], ds.compute_signals(d, None if k == "spx" else raw[k]))
              for k in INDEXES if k != index_key and k in raw]
    status = status_text(name, sig, pos, others)
    print(status, "\n")

    if not args.dry_run:
        log = signal_log.append_row(LOG_PATH, signal_log.make_row(index_key, sig, pos, target.dropna().iloc[-1],
                                                                  len(problems), now))
    else:
        log = signal_log.read_log(LOG_PATH)
    card_text = signal_log.scorecard_text(signal_log.scorecard(log, index_key, target, min_level), name)

    alerts = [Alert(f"{index_key}-{e.key}", SEVERITY[e.kind], f"[{name}] {e.title}", e.message) for e in events]
    if problems:
        alerts.append(Alert(f"data-{now:%Y-%m-%d}", "warning", "資料異常（訊號仍照算，但可能不準）",
                            "\n".join(problems)))
    if args.status:
        alerts.append(Alert(f"{index_key}-status-{sig.index[-1]:%Y-%m-%d}", "info", f"[{name}] 每日狀態", status))
    if weekly:
        alerts.append(Alert(f"{index_key}-weekly-{sig.index[-1]:%Y-%m-%d}", "info", f"[{name}] 本週摘要",
                            weekly_text(name, sig, fwd, target, evidence, min_level, status, problems, card_text)))
    for a in alerts:
        print(f"[{a.severity.upper()}] {a.title}\n{a.message}\n")
    if not alerts:
        print("今天沒有新訊號。")
        return
    if args.dry_run:
        print("(dry run，未發送)")
        return
    results = dispatch_alerts(alerts)
    if not results:
        print(f"未發送：ALERTS_ENABLED={os.environ.get('ALERTS_ENABLED', '0')}，已設定的通道：{enabled_channels() or '無'}")
    for r in results:
        print(f"{r['channel']}: {r['status']}")


if __name__ == "__main__":
    main()
