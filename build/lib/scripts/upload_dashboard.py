#!/usr/bin/env python3
"""Upload Grafana dashboard via HTTP API.

This helper imports the cache/TTL dashboard stored in ``docs/grafana`` into
Grafana. The dashboard must contain panels referencing ``lmdb_evictions_total``
and ``ingest_queue_fill_ratio`` metrics. A ``GRAFANA_URL`` and optional
``GRAFANA_TOKEN`` environment variables configure the API endpoint.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

GRAFANA_URL = os.environ.get("GRAFANA_URL", "http://localhost:3000")
GRAFANA_TOKEN = os.environ.get("GRAFANA_TOKEN")


def _contains_metric(obj: Any, metric: str) -> bool:
    """Return ``True`` if ``metric`` appears in ``obj`` recursively."""
    if isinstance(obj, dict):
        return any(_contains_metric(v, metric) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_metric(i, metric) for i in obj)
    if isinstance(obj, str):
        return metric in obj
    return False


def validate_dashboard(path: Path) -> dict[str, Any]:
    """Load dashboard JSON and verify required metric panels exist."""
    data = json.loads(path.read_text())
    for metric in ("lmdb_evictions_total", "ingest_queue_fill_ratio"):
        if not _contains_metric(data, metric):
            raise ValueError(f"missing metric {metric}")
    return data


def upload(path: Path, folder_id: int = 0) -> None:
    """Upload ``path`` to Grafana, validating its content first."""
    dashboard = validate_dashboard(path)
    payload = {"dashboard": dashboard, "folderId": folder_id, "overwrite": True}
    headers = {"Content-Type": "application/json"}
    if GRAFANA_TOKEN:
        headers["Authorization"] = f"Bearer {GRAFANA_TOKEN}"
    url = f"{GRAFANA_URL.rstrip('/')}/api/dashboards/db"
    resp = requests.post(url, json=payload, headers=headers, timeout=10)
    resp.raise_for_status()


def main() -> None:  # pragma: no cover - manual execution
    path = (
        Path(__file__).resolve().parents[1] / "docs" / "grafana" / "cache_overview.json"
    )
    upload(path)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
