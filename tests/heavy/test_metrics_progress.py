import sys
import types

import datacreek.utils.metrics as metrics
import datacreek.utils.progress as prog


class DummyProgress:
    def __init__(self):
        self.args = None
        self.started = False
        self.stopped = False

    def add_task(self, desc, total):
        self.args = (desc, total)
        return 99

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def test_push_metrics_error(monkeypatch):
    class BadClient:
        def __init__(self, *a, **k):
            pass

        def gauge(self, k, v):
            raise RuntimeError()

    mod = types.SimpleNamespace(StatsClient=BadClient)
    monkeypatch.setitem(sys.modules, "statsd", mod)
    metrics.push_metrics({"a": 1.0})


def test_progress_context_runs(monkeypatch):
    dummy = DummyProgress()
    monkeypatch.setattr(prog, "Progress", lambda *a, **k: dummy)
    with prog.progress_context("work", 2) as (p, tid):
        assert p is dummy
        assert tid == 99
        assert dummy.started
    assert dummy.stopped
