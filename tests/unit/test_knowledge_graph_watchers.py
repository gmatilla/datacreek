import builtins
from pathlib import Path

import pytest

from datacreek.core import knowledge_graph as kg


class DummyObserver:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.joined = False

    def schedule(self, handler, path, recursive=False):
        self.handler = handler
        self.path = path
        self.recursive = recursive

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def join(self, timeout=None):
        self.joined = True


class DummyEvent:
    def __init__(self, path):
        self.src_path = str(path)


def test_cleanup_watcher(monkeypatch, tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    kg.stop_cleanup_watcher()
    cfg_file.write_text("cleanup:\n  tau: 2\n")
    monkeypatch.setattr(kg, "Observer", lambda: DummyObserver())
    monkeypatch.setattr(kg, "_DummyObserver", DummyObserver)
    monkeypatch.setattr(kg, "load_config", lambda: {"cleanup": {"tau": 2}})
    monkeypatch.setitem(kg.__dict__, "yaml", None)
    kg.start_cleanup_watcher(cfg_file)
    assert isinstance(kg._observer, DummyObserver)
    event = DummyEvent(cfg_file)
    kg._observer.handler.on_modified(event)
    assert kg.get_cleanup_cfg()["tau"] == 2
    kg.stop_cleanup_watcher()
    assert kg._observer is None


def test_verify_thresholds(monkeypatch):
    monkeypatch.setattr(kg, "load_config", lambda: {"cleanup": {"tau": 3}})
    kg.apply_cleanup_config()
    kg.CleanupConfig.tau = 3
    kg.verify_thresholds()
    monkeypatch.setattr(kg, "load_config", lambda: {"cleanup": {"tau": 4}})
    with pytest.raises(RuntimeError):
        kg.verify_thresholds()
