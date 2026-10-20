"""Daily early-warning / entry-signal check. Run once per trading day after the US close.

Usage:
    python scripts/daily_alert.py            # print status, send alerts only when a signal changes
    python scripts/daily_alert.py --status   # also send today's status even if nothing changed
    python scripts/daily_alert.py --dry-run  # print what would be sent, send nothing

Channels come from environment variables (or a local .env file); see .env.example.
Sending requires ALERTS_ENABLED=1.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bubble_monitor import daily_signals as ds  # noqa: E402
from bubble_monitor.alerts import Alert  # noqa: E402
from bubble_monitor.data_sources import load_many  # noqa: E402
from bubble_monitor.notifications import dispatch_alerts, enabled_channels  # noqa: E402

CACHE_DIR = ROOT / "data" / "backtest_cache"
FRED_IDS = {"hy": "BAMLH0A0HYM2", "baa10y": "BAA10Y", "t10y2y": "T10Y2Y"}
YAHOO_IDS = {"spx": "^GSPC", "rut": "^RUT", "vix_daily": "^VIX", "vix3m_daily": "^VIX3M"}
SEVERITY = {"warning_up": "warning", "warning_clear": "info", "entry": "info", "trend": "warning", "position": "warning"}


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def status_text(sig, pos) -> str:
    today = sig.iloc[-1]
    lvl = int(today["warning_level"])
    lines = [f"{sig.index[-1]:%Y-%m-%d} 美股狀態",
             f"距一年高點 {today['dd_1y']:+.1%}，VIX {today['vix']:.1f}"]
    if today["near_high"]:
        lines.append(f"早期警示：{ds.WARNING_LEVEL_NAMES[lvl]}（{lvl}/5）")
        lines += [f"  {'●' if today[f'w_{c}'] else '○'} {label}" for c, label in ds.WARNING_CONDITIONS.items()]
    else:
        lines.append("已進入修正區（跌幅 >10%），改看進場訊號：")
        lines += [f"  {'●' if today[f'e_{k}'] else '○'} {label}" for k, label in ds.ENTRY_SIGNALS.items()]
    ma10, price = pos.attrs["ma10_provisional"], pos.attrs["price"]
    lines.append(f"10 個月均線：上月底判定{'在上方' if pos['trend'].iloc[-1] >= 1 else '在下方'}；"
                 f"目前價格距均線 {price / ma10 - 1:+.1%}（月底收盤才確定）")
    lines.append(f"規則部位（均線 × 警示 + 恐慌進場）：{pos['target'].iloc[-1]:.0%}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="send today's status even without a new signal")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")

    raw = load_many(FRED_IDS, YAHOO_IDS, CACHE_DIR, max_age_seconds=6 * 3600)
    d = ds.build_daily_frame(raw)
    sig = ds.compute_signals(d)
    fwd = ds.forward_outcomes(d["spx"])

    # GitHub Actions passes unset variables as empty strings, so fall back on "" too.
    min_level = int(os.environ.get("ALERT_MIN_WARNING_LEVEL") or "3")
    events = ds.todays_events(sig, fwd, min_warning_level=min_level)
    pos = ds.position_frame(d["spx"], sig)
    events += ds.position_events(pos, f"{sig.index[-1]:%Y-%m-%d}")
    status = status_text(sig, pos)
    print(status, "\n")

    alerts = [Alert(e.key, SEVERITY[e.kind], e.title, e.message) for e in events]
    if args.status:
        alerts.append(Alert(f"status-{sig.index[-1]:%Y-%m-%d}", "info", "每日狀態", status))
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
