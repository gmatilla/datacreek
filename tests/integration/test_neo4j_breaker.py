import importlib.util
import sys
import types
from pathlib import Path

import pybreaker
import pytest

# Load the breaker module in isolation to avoid importing the whole package tree
mon_stub = types.SimpleNamespace(
    breaker_state=None,
    _METRICS={"breaker_state": None},
    update_metric=None,
)


def _update(name, value, labels=None):
    g = mon_stub._METRICS.get(name)
    if g is not None:
        g.set(value)


mon_stub.update_metric = _update

# mon_stub defined above


# mon_stub defined above

import unittest.mock

@pytest.fixture(scope="module")
def breaker_module():
    """Load the breaker module with monitoring mocked."""
    # Use patch.dict to safely patch sys.modules for the duration of the fixture
    with unittest.mock.patch.dict(sys.modules, {"datacreek.analysis.monitoring": mon_stub}):
        # We need to ensure we import a fresh or correct module
        # Since we are using standard import now, we might rely on it not being loaded yet or reload it
        # But for safety in tests, we can just import.
        # If it was already loaded by another test without mock, it might be an issue?
        # Ideally we want to force reload if we change dependencies.
        # But here we just want to ensure it CAN import.
        from datacreek.utils import neo4j_breaker
        importlib.reload(neo4j_breaker) # Reload to ensure it picks up the mock if needed
        yield neo4j_breaker


@pytest.fixture(autouse=True)
def reset_breaker(breaker_module):
    breaker_module.reconfigure(fail_max=1, timeout=0)
    yield
    breaker_module.neo4j_breaker.close()




def test_breaker_metric_and_recovery(monkeypatch, breaker_module):
    vals = []

    class DummyGauge:
        def set(self, v: float):
            vals.append(v)

    monkeypatch.setattr(monitoring, "breaker_state", DummyGauge(), raising=False)
    monkeypatch.setitem(monitoring._METRICS, "breaker_state", monitoring.breaker_state)

    def fail():
        raise RuntimeError("fail")

    with pytest.raises(pybreaker.CircuitBreakerError):
        breaker_module.neo4j_breaker.call(fail)
    assert vals[-1] == 1

    breaker_module.neo4j_breaker.call(lambda: None)
    assert vals[-1] == 0


def test_breaker_open_then_half_open(monkeypatch, breaker_module):
    events = []

    class DummyGauge:
        def set(self, v: float):
            events.append(v)

    monkeypatch.setattr(monitoring, "breaker_state", DummyGauge(), raising=False)
    monkeypatch.setitem(monitoring._METRICS, "breaker_state", monitoring.breaker_state)

    breaker_module.reconfigure(fail_max=2, timeout=0)

    def fail():
        raise RuntimeError("fail")

    with pytest.raises(RuntimeError):
        breaker_module.neo4j_breaker.call(fail)
    with pytest.raises(pybreaker.CircuitBreakerError):
        breaker_module.neo4j_breaker.call(fail)

    assert events[-1] == 1

    breaker_module.neo4j_breaker.call(lambda: None)
    assert events[-1] == 0
