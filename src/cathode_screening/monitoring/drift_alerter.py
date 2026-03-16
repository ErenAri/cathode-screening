"""
Drift alerting — sends notifications when model drift is detected.

Supports:
    - Slack webhooks
    - PagerDuty Events API v2
    - Generic webhook (POST JSON)
    - Console logging (fallback)

Configuration:
    CATHODE_DRIFT_ALERT_BACKEND=slack|pagerduty|webhook|console
    CATHODE_SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
    CATHODE_PAGERDUTY_ROUTING_KEY=...
    CATHODE_WEBHOOK_URL=https://...
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib import request as urlrequest
from urllib.error import URLError

logger = logging.getLogger(__name__)


def _post_json(url: str, payload: dict, headers: Optional[dict] = None) -> int:
    """POST JSON to a URL and return HTTP status code."""
    data = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)

    req = urlrequest.Request(url, data=data, headers=req_headers, method="POST")
    try:
        with urlrequest.urlopen(req, timeout=10) as resp:
            return resp.status
    except URLError as exc:
        logger.error("Alert POST failed: %s", exc)
        return 0


def _format_drift_message(drift_data: Dict[str, Any]) -> str:
    """Format drift data into a human-readable message."""
    drifted = drift_data.get("drifted_columns", [])
    columns = drift_data.get("columns", {})

    lines = [
        "**Model Drift Detected — CathodeScreen**",
        f"Time: {datetime.now(timezone.utc).isoformat()}",
        f"Drifted columns: {', '.join(drifted)}",
        "",
    ]

    for col in drifted:
        stats = columns.get(col, {})
        psi = stats.get("psi", "N/A")
        lines.append(f"  - `{col}`: PSI = {psi}")

    if drift_data.get("retrain_recommended"):
        lines.append("")
        lines.append("**Action required**: Retrain recommended. Run `dvc repro train`.")

    return "\n".join(lines)


def send_slack_alert(drift_data: Dict[str, Any]) -> None:
    """Send drift alert to Slack via incoming webhook."""
    url = os.getenv("CATHODE_SLACK_WEBHOOK_URL", "")
    if not url:
        logger.warning("CATHODE_SLACK_WEBHOOK_URL not set, skipping Slack alert")
        return

    message = _format_drift_message(drift_data)
    payload = {
        "text": message,
        "blocks": [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": message},
            }
        ],
    }

    status = _post_json(url, payload)
    if status == 200:
        logger.info("Slack drift alert sent successfully")
    else:
        logger.error("Slack drift alert failed with status %d", status)


def send_pagerduty_alert(drift_data: Dict[str, Any]) -> None:
    """Send drift alert to PagerDuty Events API v2."""
    routing_key = os.getenv("CATHODE_PAGERDUTY_ROUTING_KEY", "")
    if not routing_key:
        logger.warning("CATHODE_PAGERDUTY_ROUTING_KEY not set, skipping PagerDuty alert")
        return

    drifted = drift_data.get("drifted_columns", [])
    severity = "warning" if len(drifted) < 3 else "error"

    payload = {
        "routing_key": routing_key,
        "event_action": "trigger",
        "dedup_key": f"cathode-drift-{'-'.join(sorted(drifted))}",
        "payload": {
            "summary": f"CathodeScreen model drift detected in {len(drifted)} column(s): {', '.join(drifted)}",
            "severity": severity,
            "source": "cathode-screening",
            "component": "ml-model",
            "group": "drift-detection",
            "class": "model-drift",
            "custom_details": drift_data,
        },
    }

    status = _post_json("https://events.pagerduty.com/v2/enqueue", payload)
    if status in (200, 202):
        logger.info("PagerDuty drift alert sent successfully")
    else:
        logger.error("PagerDuty drift alert failed with status %d", status)


def send_webhook_alert(drift_data: Dict[str, Any]) -> None:
    """Send drift alert to a generic webhook."""
    url = os.getenv("CATHODE_WEBHOOK_URL", "")
    if not url:
        logger.warning("CATHODE_WEBHOOK_URL not set, skipping webhook alert")
        return

    payload = {
        "event": "model_drift",
        "service": "cathode-screening",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": drift_data,
    }

    status = _post_json(url, payload)
    if status in range(200, 300):
        logger.info("Webhook drift alert sent to %s (status %d)", url, status)
    else:
        logger.error("Webhook drift alert failed: status %d", status)


def send_console_alert(drift_data: Dict[str, Any]) -> None:
    """Log drift alert to console (fallback)."""
    message = _format_drift_message(drift_data)
    logger.warning("DRIFT ALERT:\n%s", message)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_ALERT_BACKENDS = {
    "slack": send_slack_alert,
    "pagerduty": send_pagerduty_alert,
    "webhook": send_webhook_alert,
    "console": send_console_alert,
}


def send_drift_alert(drift_data: Dict[str, Any]) -> None:
    """Send drift alert via the configured backend.

    Reads CATHODE_DRIFT_ALERT_BACKEND env var (default: "console").
    Supports comma-separated backends for multi-channel alerting:
        CATHODE_DRIFT_ALERT_BACKEND=slack,pagerduty
    """
    raw = os.getenv("CATHODE_DRIFT_ALERT_BACKEND", "console")
    backends = [b.strip().lower() for b in raw.split(",") if b.strip()]

    if not backends:
        backends = ["console"]

    for backend_name in backends:
        handler = _ALERT_BACKENDS.get(backend_name)
        if handler is None:
            logger.warning("Unknown drift alert backend: %s", backend_name)
            continue
        try:
            handler(drift_data)
        except Exception as exc:
            logger.error("Drift alert via %s failed: %s", backend_name, exc)
