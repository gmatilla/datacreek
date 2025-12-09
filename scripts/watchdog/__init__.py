#!/usr/bin/env python3
"""Cron watchdog checking ANN latency, eigsh timeouts and disk usage."""

from __future__ import annotations

import json
import os
import shutil
from typing import Dict, List

import requests

METRICS_URL = os.environ.get("METRICS_URL", "http://localhost:8000/metrics")
from tempfile import gettempdir

# Use the OS temporary directory for storing the watchdog state file
# to avoid hardcoding a specific path.
STATE_FILE = os.environ.get(
    "WATCHDOG_STATE", os.path.join(gettempdir(), "datacreek_watchdog.json")
)


def parse_metrics(text: str) -> Dict[str, float]:
    """Return mapping of metric name to value from Prometheus text."""
    values: Dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                values[parts[0]] = float(parts[-1])
            except ValueError:
                continue
    return values


def check_alerts(
    current: Dict[str, float], previous: Dict[str, float], disk_pct: float
) -> List[str]:
    """Return list of alert keys triggered by metric thresholds."""
    alerts: List[str] = []
    if (
        current.get("eigsh_timeouts_total", 0) - previous.get("eigsh_timeouts_total", 0)
        > 10
    ):
        alerts.append("eigsh_timeouts_total")
    bucket = current.get('ann_latency_seconds_bucket{le="2"}', 0)
    total = current.get("ann_latency_seconds_count", 1)
    if total and bucket / total < 0.95:
        alerts.append("ann_latency")
    if disk_pct > 85:
        alerts.append("disk_usage")
    return alerts


def main() -> None:  # pragma: no cover - integration
    resp = requests.get(METRICS_URL, timeout=5)
    metrics = parse_metrics(resp.text)
    prev: Dict[str, float] = {}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as fh:
                prev = json.load(fh)
        except Exception:
            prev = {}
    usage = shutil.disk_usage(".")
    pct = 100 * usage.used / usage.total
    alerts = check_alerts(metrics, prev, pct)
    for a in alerts:
        print("ALERT:", a)
    with open(STATE_FILE, "w") as fh:
        json.dump({"eigsh_timeouts_total": metrics.get("eigsh_timeouts_total", 0)}, fh)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
