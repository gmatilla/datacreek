import json
import logging
import pickle
import time

import pytest

pytest.importorskip("lmdb")
import lmdb

pytest.importorskip("networkx")

from datacreek.analysis import mapper
from datacreek.utils.evict_log import clear_eviction_logs, evict_logs, log_eviction


def test_eviction_log_ring_buffer():
    clear_eviction_logs()
    for i in range(105_000):
        log_eviction(f"k{i}", time.time(), "ttl")
    assert len(evict_logs) == 100_000
    assert evict_logs[0].key == "k5000"


def test_manual_eviction(tmp_path):
    env = lmdb.open(str(tmp_path / "db"), map_size=1 << 20)
    with env.begin(write=True) as txn:
        txn.put(b"k0", pickle.dumps((time.time(), b"d")))
    clear_eviction_logs()
    mapper.delete_l2_entry(env, "k0")
    assert any(log.cause == "manual" and log.key == "k0" for log in evict_logs)
    env.close()


def test_eviction_metrics(monkeypatch):
    counter_calls = []
    gauge_values = []

    class DummyCounter:
        def labels(self, *, cause, type):  # type: ignore[override]
            def inc():
                counter_calls.append((cause, type))

            return type("L", (), {"inc": inc})

    class DummyGauge:
        def labels(self, *, cause):
            def set(value):
                gauge_values.append((cause, value))

            return type("L", (), {"set": set})

    monkeypatch.setattr(
        "datacreek.analysis.monitoring.lmdb_evictions_total",
        DummyCounter(),
        raising=False,
    )
    monkeypatch.setattr(
        "datacreek.analysis.monitoring.lmdb_eviction_last_ts",
        DummyGauge(),
        raising=False,
    )
    clear_eviction_logs()
    log_eviction("img:k1", 42.0, "quota")
    assert counter_calls == [("quota", "img")]
    assert gauge_values == [("quota", 42.0)]


def test_eviction_json_logging(caplog):
    caplog.set_level(logging.INFO, logger="datacreek.eviction")
    clear_eviction_logs()
    log_eviction("kj", 123.0, "ttl")
    assert any("lmdb_eviction" in r.message for r in caplog.records)
    data = json.loads(caplog.records[0].message)
    assert data["cause"] == "ttl"
