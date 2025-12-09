import json
import os

import datacreek.utils.backpressure as bp


def test_acquire_and_release(monkeypatch):
    calls = []
    monkeypatch.setattr(bp.monitoring, "update_metric", lambda *a, **k: calls.append(a))
    bp.set_limit(2)
    assert bp.active_count() == 0
    assert bp.has_capacity() is True
    assert bp.acquire_slot() is True
    assert bp.active_count() == 1
    assert bp.has_capacity() is True
    assert bp.acquire_slot() is True
    assert bp.has_capacity() is False
    assert bp.active_count() == 2
    assert bp.acquire_slot() is False
    bp.release_slot()
    assert bp.active_count() == 1
    assert bp.has_capacity() is True
    # metrics should have been updated
    assert len(calls) > 0


def test_acquire_with_backoff(tmp_path, monkeypatch):
    bp.set_limit(1)
    assert bp.acquire_slot() is True
    monkeypatch.setattr(bp.time, "sleep", lambda s: None)
    spool = tmp_path
    data = {"a": 1}
    result = bp.acquire_slot_with_backoff(
        retries=1, base_delay=0, spool_dir=str(spool), spool_data=data
    )
    assert result is False
    files = list(spool.iterdir())
    assert len(files) == 1
    with open(files[0], "r", encoding="utf-8") as f:
        assert json.load(f) == data
    # cleanup
    bp.release_slot()
