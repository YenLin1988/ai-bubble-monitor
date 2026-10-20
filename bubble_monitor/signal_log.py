"""Forward-test log: what the system said each day, and how it turned out.

Backtests can be fooled by hindsight; this log cannot. Each daily run appends the
signals exactly as they were computed that day (so later data revisions or rule
changes do not rewrite history). The scorecard then grades every logged warning
and entry signal against prices that arrived afterwards.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from .daily_signals import ENTRY_SIGNALS, WARNING_CONDITIONS, episode_starts

COLUMNS = (["date", "index", "price", "dd_1y", "warning_level"]
           + [f"w_{c}" for c in WARNING_CONDITIONS]
           + [f"e_{k}" for k in ENTRY_SIGNALS]
           + ["trend", "target", "data_problems", "logged_at"])

WARNING_HORIZON = 126  # trading days (~6 months) to judge a warning
ENTRY_HORIZONS = (63, 252)


def make_row(index_key: str, sig: pd.DataFrame, pos: pd.DataFrame, price: float, n_problems: int,
             now: dt.datetime | None = None) -> dict:
    today = sig.iloc[-1]
    row = {"date": sig.index[-1].strftime("%Y-%m-%d"), "index": index_key, "price": round(float(price), 2),
           "dd_1y": round(float(today["dd_1y"]), 4), "warning_level": int(today["warning_level"])}
    row.update({f"w_{c}": int(bool(today[f"w_{c}"])) for c in WARNING_CONDITIONS})
    row.update({f"e_{k}": int(bool(today[f"e_{k}"])) for k in ENTRY_SIGNALS})
    row.update({"trend": float(pos["trend"].iloc[-1]), "target": float(pos["target"].iloc[-1]),
                "data_problems": n_problems,
                "logged_at": (now or dt.datetime.now(dt.timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")})
    return row


def read_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype={"date": str, "index": str})


def append_row(path: Path, row: dict) -> pd.DataFrame:
    """Add today's row; a rerun on the same date replaces that date's row for the index."""
    log = read_log(path)
    log = log[~((log["date"] == row["date"]) & (log["index"] == row["index"]))]
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)[COLUMNS]
    log = log.sort_values(["index", "date"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(path, index=False)
    return log


def scorecard(log: pd.DataFrame, index_key: str, prices: pd.Series, min_level: int = 3) -> dict:
    """Grade logged signals for one index against prices observed after each signal."""
    sub = log[log["index"] == index_key].copy()
    out = {"days_logged": len(sub), "first_day": sub["date"].min() if len(sub) else None,
           "warnings": [], "entries": []}
    if not len(sub):
        return out
    sub.index = pd.to_datetime(sub["date"])
    p = prices.dropna().sort_index()

    def after(date, n):
        seg = p.loc[date:]
        return seg.iloc[: n + 1], len(seg) - 1 >= n

    near = sub["dd_1y"] > -0.10
    for date in episode_starts((sub["warning_level"] >= min_level) & near):
        seg, done = after(date, WARNING_HORIZON)
        mdd = float((seg / seg.cummax() - 1).min()) if len(seg) > 1 else np.nan
        out["warnings"].append({"date": date.date(), "level": int(sub.at[date, "warning_level"]),
                                "max_drawdown_so_far": mdd, "complete": done,
                                "hit10": mdd <= -0.10 if not np.isnan(mdd) else False})
    for name in ENTRY_SIGNALS:
        for date in episode_starts(sub[f"e_{name}"].astype(bool)):
            row = {"date": date.date(), "signal": name}
            for h in ENTRY_HORIZONS:
                seg, done = after(date, h)
                row[f"ret_{h}"] = float(seg.iloc[-1] / seg.iloc[0] - 1) if done else np.nan
            seg, _ = after(date, 126)
            row["lowest_so_far"] = float(seg.min() / seg.iloc[0] - 1)
            out["entries"].append(row)
    return out


def scorecard_text(card: dict, index_name: str) -> str:
    if not card["days_logged"]:
        return f"前向測試（{index_name}）：還沒有紀錄。"
    lines = [f"前向測試（{index_name}，{card['first_day']} 起共 {card['days_logged']} 個交易日的即時紀錄）："]
    w = card["warnings"]
    if w:
        done = [x for x in w if x["complete"]]
        hits = sum(x["hit10"] for x in w)
        lines.append(f"  早期警告 {len(w)} 次：已有 {hits} 次之後跌 ≥10%"
                     + (f"（已滿 6 個月的 {len(done)} 次中 {sum(x['hit10'] for x in done)} 次）" if done else "（都還沒滿 6 個月）"))
        for x in w[-3:]:
            status = "已滿 6 個月" if x["complete"] else "觀察中"
            lines.append(f"    {x['date']} 等級 {x['level']}：之後最大回撤 {x['max_drawdown_so_far']:+.1%}（{status}）")
    else:
        lines.append("  早期警告：還沒有新的警告紀錄。")
    e = card["entries"]
    if e:
        for x in e[-3:]:
            r = "、".join(f"{h // 21} 個月後 {x[f'ret_{h}']:+.1%}" for h in ENTRY_HORIZONS if not np.isnan(x[f"ret_{h}"]))
            lines.append(f"  進場訊號 {x['date']}（{x['signal']}）：之後最低 {x['lowest_so_far']:+.1%}"
                         + (f"；{r}" if r else "；觀察中"))
    else:
        lines.append("  進場訊號：還沒有紀錄。")
    return "\n".join(lines)
