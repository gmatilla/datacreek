import types

import pytest

import datacreek.utils.progress as prog


class DummyProgress:
    def __init__(self):
        self.args = None
        self.started = False
        self.stopped = False

    def add_task(self, desc, total):
        self.args = (desc, total)
        return 42

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def test_create_progress(monkeypatch):
    dummy = DummyProgress()
    monkeypatch.setattr(prog, "Progress", lambda *a, **k: dummy)
    prog_inst, tid = prog.create_progress("desc", 3)
    assert prog_inst is dummy
    assert tid == 42
    assert dummy.args == ("desc", 3)


def test_progress_context(monkeypatch):
    dummy = DummyProgress()
    monkeypatch.setattr(prog, "Progress", lambda *a, **k: dummy)
    with prog.progress_context("run", 1) as (p, tid):
        assert p is dummy
        assert tid == 42
        assert dummy.started
    assert dummy.stopped


def test_progress_context_exception(monkeypatch):
    dummy = DummyProgress()
    monkeypatch.setattr(prog, "Progress", lambda *a, **k: dummy)
    with pytest.raises(RuntimeError):
        with prog.progress_context("oops", 2):
            assert dummy.started
            raise RuntimeError("fail")
    assert dummy.stopped
