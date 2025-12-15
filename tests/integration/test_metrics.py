import importlib.util
import sys
import types
from pathlib import Path

import pytest

try:
    import datacreek.api  # noqa: F401
except Exception:
    pytest.skip("datacreek.api unavailable", allow_module_level=True)

from datacreek.utils.metrics import push_metrics


def test_push_metrics_no_statsd(monkeypatch):
    # Ensure function does nothing when statsd is missing
    monkeypatch.setitem(sys.modules, "statsd", None, raising=False)
    push_metrics({"a": 1.0})


def test_push_metrics_with_client(monkeypatch):
    calls = []

    class Dummy:
        def __init__(self, *_, **__):
            pass

        def gauge(self, key, value):
            calls.append((key, value))

    monkeypatch.setitem(sys.modules, "statsd", types.SimpleNamespace(StatsClient=Dummy))

    push_metrics({"a": 2.0}, prefix="test")
    assert calls == [("a", 2.0)]
