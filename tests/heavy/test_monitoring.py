import importlib
import sys
import types

import pytest


class DummyGauge:
    def __init__(self, *a, **k):
        self.value = None
        self.registry = k.get("registry")
        self.labelnames = k.get("labelnames", [])

    def set(self, v):
        self.value = v

    def labels(self, **labels):
        return self


class DummyCounter(DummyGauge):
    def inc(self):
        pass


@pytest.fixture()
def reload_monitoring(monkeypatch):
    metrics = types.SimpleNamespace(
        REGISTRY=types.SimpleNamespace(_names_to_collectors={}),
        CollectorRegistry=lambda: None,
        Counter=DummyCounter,
        Gauge=DummyGauge,
        push_to_gateway=lambda gateway, job, registry: pushes.append((gateway, job)),
        start_http_server=lambda port: started.append(port),
    )
    monkeypatch.setitem(sys.modules, "prometheus_client", metrics)
    monkeypatch.setitem(
        sys.modules,
        "datacreek.utils.config",
        types.SimpleNamespace(load_config=lambda: {}),
    )
    if "datacreek.analysis.monitoring" in sys.modules:
        del sys.modules["datacreek.analysis.monitoring"]
    mod = importlib.import_module("datacreek.analysis.monitoring")
    started.clear()
    pushes.clear()
    mod._SERVER_STARTED = False
    return mod


started = []
pushes = []


@pytest.mark.heavy
def test_start_server_idempotent(reload_monitoring):
    mod = reload_monitoring
    mod.start_metrics_server(1234)
    mod.start_metrics_server(1234)
    assert started == [1234]
    assert mod._SERVER_STARTED


@pytest.mark.heavy
def test_push_and_update(monkeypatch, reload_monitoring):
    mod = reload_monitoring
    mod.push_metrics_gateway({"foo": 1.5}, "host:123")
    assert pushes == [("host:123", "datacreek")]
    g = DummyGauge()
    mod._METRICS["bar"] = g
    mod.update_metric("bar", 0.7)
    assert g.value == 0.7
    g2 = DummyGauge()
    g2.labels = lambda **kw: g2
    mod._METRICS["baz"] = g2
    mod.update_metric("baz", 2.0, labels={"x": "1"})
    assert g2.value == 2.0
