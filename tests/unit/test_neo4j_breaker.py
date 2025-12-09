import types

from datacreek.utils import neo4j_breaker as nb


def test_listener_updates(monkeypatch):
    calls = []
    monkeypatch.setattr(
        nb.monitoring, "update_metric", lambda n, v: calls.append((n, v))
    )
    listener = nb._PrometheusListener()
    listener.state_change(None, None, types.SimpleNamespace(name="open"))
    listener.state_change(None, None, types.SimpleNamespace(name="closed"))
    assert calls == [("breaker_state", 1), ("breaker_state", 0)]


def test_reconfigure(monkeypatch):
    calls = []
    monkeypatch.setattr(
        nb.monitoring, "update_metric", lambda n, v: calls.append((n, v))
    )
    nb.reconfigure(fail_max=2, timeout=10)
    assert nb.neo4j_breaker.fail_max == 2
    assert nb.neo4j_breaker.reset_timeout == 10
    assert calls[-1] == ("breaker_state", 0)
