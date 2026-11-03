#!/usr/bin/env python3
"""Health watchdog — polls /ops/health, emails on state transitions.

Runs from a launchd timer (com.aitrader.health-watchdog) every 5 minutes on
this machine. Logic:

  1. GET http://127.0.0.1:9000/ops/health  (localhost → no auth)
  2. Compare overall status against the last-seen status in the state file.
  3. On a TRANSITION (green→red, red→green, etc.) send one email to
     ALERT_EMAIL via the same SMTP config the app already uses (.env
     AITRADER_SMTP_*). No transition → no email, so a broken system alerts once
     instead of every 5 minutes, and you also get the "recovered" mail.

Special case: if /ops/health itself is unreachable (9000 down), that is
treated as status="down" — the most severe state of all.

State file: ~/.cache/aitrader/health_watchdog_state.json
Log:        stdout (launchd redirects to ~/Library/Logs/com.aitrader.health-watchdog.out.log)
"""
from __future__ import annotations

import json
import os
import smtplib
import ssl
import sys
import urllib.request
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

HEALTH_URL = os.environ.get("WATCHDOG_HEALTH_URL", "http://127.0.0.1:9000/ops/health")
ALERT_EMAIL = os.environ.get("WATCHDOG_ALERT_EMAIL", "")
STATE_FILE = Path.home() / ".cache" / "aitrader" / "health_watchdog_state.json"

# Severity order for transition wording
_SEV = {"green": 0, "yellow": 1, "red": 2, "down": 3}


def load_env(repo_root: Path) -> None:
    env_file = repo_root / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def fetch_health() -> dict:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"status": "down", "checks": {}, "error": str(exc)[:300]}


def read_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def write_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=1))


def bad_checks_summary(health: dict) -> str:
    lines = []
    for name, c in (health.get("checks") or {}).items():
        st = c.get("status", "?")
        if st != "green":
            detail = {k: v for k, v in c.items() if k != "status"}
            lines.append(f"  [{st.upper()}] {name}: {json.dumps(detail, ensure_ascii=False)[:300]}")
    if health.get("error"):
        lines.append(f"  [DOWN] /ops/health unreachable: {health['error']}")
    return "\n".join(lines) or "  (all green)"


def send_alert(subject: str, body: str) -> bool:
    host = os.environ.get("AITRADER_SMTP_HOST")
    port = int(os.environ.get("AITRADER_SMTP_PORT", "465"))
    user = os.environ.get("AITRADER_SMTP_USER")
    password = os.environ.get("AITRADER_SMTP_PASSWORD")
    use_ssl = os.environ.get("AITRADER_SMTP_USE_SSL", "1") not in ("0", "false", "")
    if not (host and user and password):
        print("watchdog: SMTP not configured; would have sent:", subject)
        return False
    msg = EmailMessage()
    msg["From"] = f"AITrader Watchdog <{user}>"
    msg["To"] = ALERT_EMAIL
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        if use_ssl:
            with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context(), timeout=20) as s:
                s.login(user, password)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=20) as s:
                s.starttls(context=ssl.create_default_context())
                s.login(user, password)
                s.send_message(msg)
        print(f"watchdog: alert sent → {ALERT_EMAIL}: {subject}")
        return True
    except Exception as exc:
        print(f"watchdog: SMTP send failed: {exc}")
        return False


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    load_env(repo_root)

    health = fetch_health()
    now_status = health.get("status", "down")
    state = read_state()
    prev_status = state.get("status", "green")
    now_iso = datetime.now(timezone.utc).isoformat()

    print(f"watchdog: {prev_status} → {now_status}")

    if now_status != prev_status:
        direction = "恶化" if _SEV.get(now_status, 3) > _SEV.get(prev_status, 0) else "恢复"
        emoji = {"green": "✅", "yellow": "⚠️", "red": "🔴", "down": "💀"}.get(now_status, "❓")
        subject = f"{emoji} AITrader 系统状态{direction}: {prev_status} → {now_status}"
        body = (
            f"时间: {now_iso}\n"
            f"状态变化: {prev_status} → {now_status}\n\n"
            f"非绿项:\n{bad_checks_summary(health)}\n\n"
            f"面板: http://127.0.0.1:9000/static/ops.html\n"
        )
        send_alert(subject, body)

    write_state({"status": now_status, "ts": now_iso})
    return 0


if __name__ == "__main__":
    sys.exit(main())
