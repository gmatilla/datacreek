import importlib.abc
import importlib.util
import sys
import types
from pathlib import Path

stub = types.ModuleType("requests")
stub.post = lambda *a, **k: None
sys.modules["requests"] = stub
import json
import os

spec = importlib.util.spec_from_file_location(
    "upload_dashboard",
    Path(__file__).resolve().parents[1] / "scripts" / "upload_dashboard.py",
)
os.environ["GRAFANA_URL"] = "http://grafana"
os.environ["GRAFANA_TOKEN"] = "tok"
upload_dashboard = importlib.util.module_from_spec(spec)
assert isinstance(spec.loader, importlib.abc.Loader)
spec.loader.exec_module(upload_dashboard)


def test_validate_dashboard_metrics():
    path = (
        Path(__file__).resolve().parents[1] / "docs" / "grafana" / "cache_overview.json"
    )
    data = upload_dashboard.validate_dashboard(path)
    text = json.dumps(data)
    assert "lmdb_evictions_total" in text
    assert "ingest_queue_fill_ratio" in text


def test_upload_dashboard(monkeypatch):
    calls = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls["url"] = url
        calls["json"] = json
        calls["headers"] = headers

        class R:
            def raise_for_status(self):
                pass

        return R()

    monkeypatch.setattr(upload_dashboard.requests, "post", fake_post)
    path = (
        Path(__file__).resolve().parents[1] / "docs" / "grafana" / "cache_overview.json"
    )
    upload_dashboard.upload(path)
    assert calls["url"] == "http://grafana/api/dashboards/db"
    assert calls["headers"]["Authorization"] == "Bearer tok"
    payload_text = json.dumps(calls["json"])
    assert "lmdb_evictions_total" in payload_text
    assert "ingest_queue_fill_ratio" in payload_text
