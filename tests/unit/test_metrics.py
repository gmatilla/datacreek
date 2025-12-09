import sys
import types

from datacreek.utils import metrics


class FakeClient:
    instance = None

    def __init__(self, host, port, prefix):
        self.info = (host, port, prefix)
        self.recorded = {}
        FakeClient.instance = self

    def gauge(self, key, value):
        self.recorded[key] = value


def test_push_metrics(monkeypatch):
    mod = types.SimpleNamespace(StatsClient=FakeClient)
    monkeypatch.setitem(sys.modules, "statsd", mod)
    metrics.push_metrics({"a": 1.0, "b": 2.0}, prefix="p", host="h", port=123)
    client = FakeClient.instance
    assert client.info == ("h", 123, "p")
    assert client.recorded == {"a": 1.0, "b": 2.0}


def test_push_metrics_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "statsd", None)
    metrics.push_metrics({"x": 1})


def test_push_metrics_error(monkeypatch):
    class ErrClient(FakeClient):
        def gauge(self, key, value):
            raise RuntimeError("boom")

    mod = types.SimpleNamespace(StatsClient=ErrClient)
    monkeypatch.setitem(sys.modules, "statsd", mod)
    metrics.push_metrics({"z": 5})
    assert ErrClient.instance.recorded == {}
