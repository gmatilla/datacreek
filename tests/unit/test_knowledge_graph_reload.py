from types import SimpleNamespace

import pytest

from datacreek.core import knowledge_graph as kg


def test_config_reloader(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(kg, "_load_cleanup", lambda: calls.append("load"))
    monkeypatch.setattr(kg, "apply_cleanup_config", lambda: calls.append("apply"))
    cfg_path = tmp_path / "cfg.yaml"
    reloader = kg.ConfigReloader(cfg_path)
    event = SimpleNamespace(src_path=str(cfg_path))
    reloader.on_modified(event)
    assert calls == ["load", "apply"]


def test_start_stop_watcher(monkeypatch, tmp_path):
    monkeypatch.setenv("DATACREEK_CONFIG", str(tmp_path / "x.yaml"))
    monkeypatch.setattr(kg, "_load_cleanup", lambda: None)
    monkeypatch.setattr(kg, "apply_cleanup_config", lambda: None)
    calls = []
    observer = SimpleNamespace(
        schedule=lambda *a, **k: calls.append("sched"),
        start=lambda: calls.append("start"),
        stop=lambda: calls.append("stop"),
        join=lambda timeout=None: calls.append("join"),
        daemon=False,
    )
    monkeypatch.setattr(kg, "Observer", lambda: observer)
    kg._observer = None
    kg.start_cleanup_watcher(interval=0.0)
    assert kg._observer is observer
    kg.stop_cleanup_watcher()
    assert kg._observer is None
    assert calls == ["sched", "start", "stop", "join"]


def test_verify_thresholds_mismatch(monkeypatch):
    kg.CleanupConfig.tau = 1
    monkeypatch.setattr(kg, "load_config", lambda: {"cleanup": {"tau": 2}})
    with pytest.raises(RuntimeError):
        kg.verify_thresholds()
