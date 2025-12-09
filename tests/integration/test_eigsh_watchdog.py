import time

import numpy as np
import pytest

import datacreek.analysis.fractal as fractal


def test_eigsh_lmax_watchdog_timeout(monkeypatch):
    # slow eigsh to trigger timeout
    def slow_eigsh(*args, **kwargs):
        time.sleep(0.1)
        return np.array([3.0])

    monkeypatch.setattr("scipy.sparse.linalg.eigsh", slow_eigsh)
    monkeypatch.setattr(
        fractal,
        "lanczos_top_eigenvalue",
        lambda *a, **k: 3.0,
    )

    called = {"n": 0}
    gauge = {"v": 0.0}

    class DummyCounter:
        def inc(self):
            called["n"] += 1

    class DummyGauge:
        def set(self, value):
            gauge["v"] = value

    monkeypatch.setattr(
        "datacreek.analysis.monitoring.eigsh_timeouts_total",
        DummyCounter(),
        raising=False,
    )
    monkeypatch.setattr(
        "datacreek.analysis.monitoring.eigsh_last_duration",
        DummyGauge(),
        raising=False,
    )

    L = np.array([[2.0, -1.0], [-1.0, 2.0]])
    val = fractal.eigsh_lmax_watchdog(L, maxiter=5, timeout=0.01)
    assert val == pytest.approx(3.0, abs=1e-2)
    assert called["n"] == 1
    assert gauge["v"] >= 0.01


def test_eigsh_safe(monkeypatch):
    """eigsh_safe should fall back on timeout and increment counter."""

    def slow_eigsh(*args, **kwargs):
        time.sleep(0.1)
        return np.array([4.0])

    monkeypatch.setattr("scipy.sparse.linalg.eigsh", slow_eigsh)
    monkeypatch.setattr(fractal, "lanczos_top_eigenvalue", lambda *a, **k: 4.0)

    called = {"n": 0}

    class DummyCounter:
        def inc(self):
            called["n"] += 1

    monkeypatch.setattr(
        "datacreek.analysis.monitoring.eigsh_timeouts_total",
        DummyCounter(),
        raising=False,
    )
    monkeypatch.setattr(
        "datacreek.analysis.monitoring.eigsh_last_duration",
        None,
        raising=False,
    )

    monkeypatch.setattr(
        "datacreek.utils.config.load_config",
        lambda: {"spectral": {"eig_timeout": 0.01}},
    )

    L = np.array([[2.0, -1.0], [-1.0, 2.0]])
    val = fractal.eigsh_safe(L)
    assert val == 4.0
    assert called["n"] == 1
