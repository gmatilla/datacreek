import pickle
import time

import lmdb

import datacreek.analysis.mapper as mapper


def test_l2_eviction_metric(tmp_path, monkeypatch):
    called = {"n": 0}

    class Dummy:
        def inc(self):
            called["n"] += 1

    monkeypatch.setattr(
        "datacreek.analysis.monitoring.redis_evictions_l2_total",
        Dummy(),
        raising=False,
    )

    env = lmdb.open(str(tmp_path / "db"), map_size=1 << 20)
    with env.begin(write=True) as txn:
        for i in range(3):
            txn.put(f"k{i}".encode(), pickle.dumps((time.time() - 3600 * 25, b"d")))
    mapper._l2_evict_once(env, limit_mb=1, ttl_h=24)
    assert called["n"] >= 1
    env.close()


def test_l2_eviction_debug_log(tmp_path, caplog):
    env = lmdb.open(str(tmp_path / "db"), map_size=1 << 20)
    with env.begin(write=True) as txn:
        txn.put(b"k0", pickle.dumps((time.time() - 3600 * 25, b"d")))
    caplog.set_level("DEBUG")
    mapper._l2_evict_once(env, limit_mb=1, ttl_h=24)
    assert any("LMDB-EVICT" in r.message for r in caplog.records)
    env.close()
