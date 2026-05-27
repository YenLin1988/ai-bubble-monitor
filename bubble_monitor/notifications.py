"""Optional notification dispatchers.

Telegram is intentionally not implemented for this project.
"""

from __future__ import annotations

import json
import os
import smtplib
import urllib.request
from email.message import EmailMessage
from typing import Iterable

from .alerts import Alert


def enabled_channels(env: dict[str, str] | None = None) -> list[str]:
    env = env or os.environ
    channels = []
    if env.get("ALERT_EMAIL_SMTP_HOST") and env.get("ALERT_EMAIL_TO"):
        channels.append("email")
    if env.get("ALERT_DISCORD_WEBHOOK_URL"):
        channels.append("discord")
    if env.get("ALERT_WEBHOOK_URL"):
        channels.append("webhook")
    return channels


def dispatch_alerts(alerts: Iterable[Alert], env: dict[str, str] | None = None) -> list[dict[str, str]]:
    env = env or os.environ
    alerts = list(alerts)
    if env.get("ALERTS_ENABLED") != "1" or not alerts:
        return []

    results: list[dict[str, str]] = []
    message = format_alert_message(alerts)

    if "email" in enabled_channels(env):
        try:
            results.append(_send_email(message, env))
        except Exception as exc:
            results.append({"channel": "email", "status": f"error: {exc}"})
    if "discord" in enabled_channels(env):
        try:
            results.append(_post_json(env["ALERT_DISCORD_WEBHOOK_URL"], {"content": message}, "discord"))
        except Exception as exc:
            results.append({"channel": "discord", "status": f"error: {exc}"})
    if "webhook" in enabled_channels(env):
        try:
            results.append(_post_json(env["ALERT_WEBHOOK_URL"], {"text": message, "alerts": [a.to_dict() for a in alerts]}, "webhook"))
        except Exception as exc:
            results.append({"channel": "webhook", "status": f"error: {exc}"})

    return results


def format_alert_message(alerts: Iterable[Alert]) -> str:
    lines = ["AI Bubble Monitor alert"]
    for alert in alerts:
        lines.append(f"[{alert.severity.upper()}] {alert.title}: {alert.message}")
    return "\n".join(lines)


def _send_email(message: str, env: dict[str, str]) -> dict[str, str]:
    msg = EmailMessage()
    msg["Subject"] = env.get("ALERT_EMAIL_SUBJECT", "AI Bubble Monitor alert")
    msg["From"] = env.get("ALERT_EMAIL_FROM", env.get("ALERT_EMAIL_USER", "ai-bubble-monitor@localhost"))
    msg["To"] = env["ALERT_EMAIL_TO"]
    msg.set_content(message)

    host = env["ALERT_EMAIL_SMTP_HOST"]
    port = int(env.get("ALERT_EMAIL_SMTP_PORT", "587"))
    user = env.get("ALERT_EMAIL_USER")
    password = env.get("ALERT_EMAIL_PASSWORD")
    use_tls = env.get("ALERT_EMAIL_TLS", "1") != "0"

    with smtplib.SMTP(host, port, timeout=20) as smtp:
        if use_tls:
            smtp.starttls()
        if user and password:
            smtp.login(user, password)
        smtp.send_message(msg)
    return {"channel": "email", "status": "sent"}


def _post_json(url: str, payload: dict, channel: str) -> dict[str, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as response:
        status = str(response.status)
    return {"channel": channel, "status": status}
