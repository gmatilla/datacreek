import types
from pathlib import Path

import datacreek.core.knowledge_graph as kg


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


def test_start_stop_watcher_idempotent(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("cleanup:\n  tau: 1\n")
    monkeypatch.setattr(kg, "Observer", lambda: DummyObserver())
    kg.stop_cleanup_watcher()
    kg.start_cleanup_watcher(cfg)
    first = kg._observer
    kg.start_cleanup_watcher(cfg)
    assert kg._observer is first
    kg.stop_cleanup_watcher()
    assert first.stopped
    kg.stop_cleanup_watcher()
    assert kg._observer is None
