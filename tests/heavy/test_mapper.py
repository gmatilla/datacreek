import importlib
import os
import pickle
import time
import types

import fakeredis
import lmdb
import networkx as nx
import pytest

from datacreek.analysis import mapper


@pytest.mark.heavy
def test_mapper_roundtrip_and_hash(tmp_path):
    g = nx.cycle_graph(3)
    nerve, cover = mapper.mapper_nerve(g, radius=1)
    recon = mapper.inverse_mapper(nerve, cover)
    assert set(g.edges()).issubset(set(recon.edges()))
    assert mapper._hash_graph(g) == mapper._hash_graph(recon)


@pytest.mark.heavy
def test_cache_mapper_nerve_full_stack(tmp_path):
    g = nx.path_graph(4)
    client = fakeredis.FakeRedis()
    db = str(tmp_path / "cache.mdb")
    ssd = str(tmp_path / "ssd")
    nerve1, cover1 = mapper.cache_mapper_nerve(
        g, 1, redis_client=client, lmdb_path=db, ssd_dir=ssd
    )
    nerve2, cover2 = mapper.cache_mapper_nerve(
        g, 1, redis_client=client, lmdb_path=db, ssd_dir=ssd
    )
    assert nx.is_isomorphic(nerve1, nerve2)
    assert cover1 == cover2
    assert client.get(f"1_{mapper._hash_graph(g)}") is not None

    client.flushall()
    nerve3, cover3 = mapper.cache_mapper_nerve(
        g, 1, redis_client=client, lmdb_path=db, ssd_dir=ssd
    )
    assert nx.is_isomorphic(nerve1, nerve3)
    assert cover1 == cover3

    env = lmdb.open(db, readonly=False)
    env.close()
    import shutil

    shutil.rmtree(db)
    client.flushall()
    nerve4, _ = mapper.cache_mapper_nerve(
        g, 1, redis_client=client, lmdb_path=db, ssd_dir=ssd
    )
    assert nx.is_isomorphic(nerve1, nerve4)


@pytest.mark.heavy
def test_adjust_ttl_updates(monkeypatch):
    monkeypatch.setattr(
        mapper, "os", types.SimpleNamespace(getloadavg=lambda: (0, 0, 0))
    )
    start = mapper._redis_ttl = 1000
    mapper._redis_hits = 1
    mapper._redis_misses = 9
    mapper._hit_ema = 0
    mapper._last_ttl_eval = 0
    monkeypatch.setattr(mapper.time, "time", lambda: 301.0)
    mapper._adjust_ttl(None)
    assert mapper._redis_ttl < start


@pytest.mark.heavy
def test_adjust_ttl_high_load(monkeypatch):
    monkeypatch.setattr(
        mapper,
        "os",
        types.SimpleNamespace(getloadavg=lambda: (1.0,), cpu_count=lambda: 1),
    )
    start = mapper._redis_ttl = 300
    mapper._redis_hits = 10
    mapper._redis_misses = 0
    mapper._hit_ema = 0.9
    mapper._last_ttl_eval = 0
    monkeypatch.setattr(mapper.time, "time", lambda: 301.0)

    class Gauge:
        def __init__(self):
            self.value = None

        def set(self, v):
            self.value = v

    class ErrorRedis(fakeredis.FakeRedis):
        def config_set(self, *a, **k):
            raise RuntimeError

        def expire(self, *a, **k):
            raise RuntimeError

    monkeypatch.setattr(mapper, "redis", fakeredis)
    import datacreek.analysis.monitoring as mon

    monkeypatch.setattr(mon, "redis_hit_ratio", Gauge())
    client = ErrorRedis()
    mapper._adjust_ttl(client, key="a")
    assert mapper._redis_ttl >= start


@pytest.mark.heavy
def test_l2_evict_and_delete(tmp_path):
    path = str(tmp_path / "db")
    env = lmdb.open(path, map_size=1 << 20)
    old_ts = time.time() - 7200
    with env.begin(write=True) as txn:
        txn.put(b"a", pickle.dumps((old_ts, b"x")))
        txn.put(b"b", pickle.dumps((time.time(), b"y")))
    mapper._l2_evict_once(env, limit_mb=1, ttl_h=1)
    with env.begin() as txn:
        assert txn.get(b"a") is None
        assert txn.get(b"b") is not None
    assert mapper.delete_l2_entry(env, "b")
    with env.begin() as txn:
        assert txn.get(b"b") is None
    env.close()


@pytest.mark.heavy
def test_eviction_thread_lifecycle(monkeypatch, tmp_path):
    def fake_config():
        return {"cache": {"l2_max_size_mb": 1, "l2_ttl_hours": 1}}

    monkeypatch.setattr(mapper, "load_config", fake_config)
    monkeypatch.setattr(mapper, "_evict_worker", lambda *a, **k: None)
    path = tmp_path / "lmdb.mdb"
    mapper.start_l2_eviction_thread(lmdb_path=str(path), interval=0.1)
    assert mapper._evict_thread is not None
    mapper.stop_l2_eviction_thread()
    assert mapper._evict_thread is None


@pytest.mark.heavy
def test_cache_put_and_get_l2(monkeypatch, tmp_path):
    g = nx.path_graph(3)
    cover = [{0, 1}, {1, 2}]
    client = fakeredis.FakeRedis()
    lmdb_path = str(tmp_path / "cache.mdb")
    ssd_dir = str(tmp_path / "ssd")

    monkeypatch.setattr(mapper, "lmdb", None)
    monkeypatch.setattr(
        mapper,
        "load_config",
        lambda: {"cache": {"l2_max_size_mb": 1, "l2_ttl_hours": 1}},
    )
    monkeypatch.setattr(mapper, "start_l2_eviction_thread", lambda *a, **k: None)

    mapper._cache_put(
        "g1",
        g,
        cover,
        redis_client=client,
        lmdb_path=lmdb_path,
        ssd_dir=ssd_dir,
        ttl=5,
    )

    client.flushall()
    res = mapper._cache_get(
        "g1", redis_client=client, lmdb_path=lmdb_path, ssd_dir=ssd_dir
    )
    assert res is not None
    nerve, c = res
    assert nx.is_isomorphic(nerve, g)
    assert c == [set(x) for x in cover]


@pytest.mark.heavy
def test_cache_get_miss(tmp_path):
    client = fakeredis.FakeRedis()
    res = mapper._cache_get(
        "missing",
        redis_client=client,
        lmdb_path=str(tmp_path / "db"),
        ssd_dir=str(tmp_path / "ssd"),
    )
    assert res is None
