# AI Bubble Monitor Architecture

This project is being moved from a single-file Streamlit prototype into a maintainable monitoring application.

## Runtime Layers

| Layer | Path | Responsibility |
|---|---|---|
| Streamlit UI | `Ai_bubble_monitor.py` | Dashboard layout, charts, KPI cards, alert/data-quality display |
| Model config | `bubble_monitor/model_config.py` | Tickers, FRED series, weights, clusters, factor methodology |
| Scoring | `bubble_monitor/scoring.py` | Normalization, weighted stress, regime classification, cluster decomposition |
| Data quality | `bubble_monitor/data_quality.py` | Source-level status rows and aggregate data-health score |
| History storage | `bubble_monitor/storage.py` | SQLite stress history and alert event storage |
| Alert rules | `bubble_monitor/alerts.py` | Deterministic alert-rule evaluation |
| Notifications | `bubble_monitor/notifications.py` | Optional Email, Discord, and generic webhook dispatch |

## Alert Policy

Telegram notification is intentionally excluded.

Supported optional channels:

- Email through SMTP
- Discord webhook
- Generic JSON webhook

The dashboard always displays active alert rules. External dispatch only runs when `ALERTS_ENABLED=1`.
Repeated external dispatch is throttled by `ALERT_COOLDOWN_HOURS` and defaults to 6 hours per alert key.

## Storage

Historical stress records are now stored in `data/stress_history.sqlite3`.

The old `stress_history.json` file is still supported as a migration source. If it exists, records are imported into SQLite when the app starts.

## Model Governance

The model table in `bubble_monitor/model_config.py` is the source of truth for:

- factor names
- source references
- safe/danger thresholds
- cluster assignment
- weights

When weights or thresholds change, update the config file and add a short README note explaining why.
