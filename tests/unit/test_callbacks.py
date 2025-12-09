import sys
import types
from importlib import reload


def test_prometheus_eta_callback_updates_gauge(monkeypatch):
    class DummyGauge:
        def __init__(self, name, desc):
            self.name = name
            self.desc = desc
            self.value = None

        def set(self, value):
            self.value = value

    dummy = types.SimpleNamespace(Gauge=DummyGauge)
    monkeypatch.setitem(sys.modules, "prometheus_client", dummy)
    from training import callbacks

    reload(callbacks)
    cb = callbacks.TrainingEtaSecondsCallback(total_steps=100)
    args = types.SimpleNamespace(max_steps=100)
    state = types.SimpleNamespace(global_step=0, max_steps=100)
    control = object()
    times = iter([0.0, 1.0])
    monkeypatch.setattr(callbacks.time, "perf_counter", lambda: next(times))
    cb.on_log(args, state, control)
    state.global_step = 50
    cb.on_log(args, state, control)
    assert cb.gauge.value == 1.0
