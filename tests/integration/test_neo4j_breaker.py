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
analysis_mod = types.ModuleType("datacreek.analysis")
analysis_mod.monitoring = mon_stub
dc_mod = types.ModuleType("datacreek")
dc_mod.analysis = analysis_mod

_orig_dc = sys.modules.get("datacreek")
_orig_analysis = sys.modules.get("datacreek.analysis")
_orig_mon = sys.modules.get("datacreek.analysis.monitoring")

sys.modules["datacreek"] = dc_mod
sys.modules["datacreek"].analysis = analysis_mod
sys.modules["datacreek.analysis"] = analysis_mod
sys.modules["datacreek.analysis.monitoring"] = mon_stub
spec = importlib.util.spec_from_file_location(
    "breaker_mod",
    Path(__file__).resolve().parents[1] / "datacreek" / "utils" / "neo4j_breaker.py",
)
breaker_mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(breaker_mod)
neo4j_breaker = breaker_mod.neo4j_breaker
reconfigure = breaker_mod.reconfigure
monitoring = mon_stub


@pytest.fixture(autouse=True)
def reset_breaker():
    reconfigure(fail_max=1, timeout=0)
    yield
    neo4j_breaker.close()


@pytest.fixture(scope="module", autouse=True)
def restore_modules():
    yield
    for name, mod in [
        ("datacreek", _orig_dc),
        ("datacreek.analysis", _orig_analysis),
        ("datacreek.analysis.monitoring", _orig_mon),
    ]:
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


def test_breaker_metric_and_recovery(monkeypatch):
    vals = []

    class DummyGauge:
        def set(self, v: float):
            vals.append(v)

    monkeypatch.setattr(monitoring, "breaker_state", DummyGauge(), raising=False)
    monkeypatch.setitem(monitoring._METRICS, "breaker_state", monitoring.breaker_state)

    def fail():
        raise RuntimeError("fail")

    with pytest.raises(pybreaker.CircuitBreakerError):
        neo4j_breaker.call(fail)
    assert vals[-1] == 1

    neo4j_breaker.call(lambda: None)
    assert vals[-1] == 0


def test_breaker_open_then_half_open(monkeypatch):
    events = []

    class DummyGauge:
        def set(self, v: float):
            events.append(v)

    monkeypatch.setattr(monitoring, "breaker_state", DummyGauge(), raising=False)
    monkeypatch.setitem(monitoring._METRICS, "breaker_state", monitoring.breaker_state)

    reconfigure(fail_max=2, timeout=0)

    def fail():
        raise RuntimeError("fail")

    with pytest.raises(RuntimeError):
        neo4j_breaker.call(fail)
    with pytest.raises(pybreaker.CircuitBreakerError):
        neo4j_breaker.call(fail)

    assert events[-1] == 1

    neo4j_breaker.call(lambda: None)
    assert events[-1] == 0
