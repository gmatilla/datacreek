import os
import pickle
import time

import fakeredis
import lmdb
import networkx as nx
import numpy as np

from datacreek.analysis import mapper


def test_mapper_roundtrip_and_hash(tmp_path):
    """Cover mapping functions and graph hashing."""
    g = nx.cycle_graph(3)
    nerve, cover = mapper.mapper_nerve(g, radius=1)
    recovered = mapper.inverse_mapper(nerve, cover)
    assert set(g.edges()).issubset(set(recovered.edges()))
    # hashing identical graphs should match
    h1 = mapper._hash_graph(g)
    h2 = mapper._hash_graph(recovered)
    assert h1 == h2


def test_inverse_mapper_cross_edges():
    """Edges between overlapping clusters are reconstructed."""

    g = nx.path_graph(4)
    nerve, cover = mapper.mapper_nerve(g, radius=1)
    recon = mapper.inverse_mapper(nerve, cover)
    for u, v in g.edges():
        assert recon.has_edge(u, v)


def test_autotune_mapper_overlap_increases_silhouette():
    """Grid-searching overlap should yield higher silhouette score."""

    g = nx.path_graph(6)
    lens = lambda graph: {n: float(n) for n in graph.nodes()}

    nerve, cover, ov, score = mapper.autotune_mapper_overlap(
        g, overlaps=[0.2, 0.3, 0.4, 0.5], n_intervals=2, lens=lens
    )
    _, base_cover = mapper.mapper_full(g, lens=lens, cover=(2, 0.5))
    base_score = mapper._cover_silhouette(lens(g), base_cover)
    assert score - base_score >= 0.03
    assert ov in {0.2, 0.3}


def test_tune_overlap_logs_metric(monkeypatch):
    """`tune_overlap` exposes chosen overlap via metric logger."""

    g = nx.path_graph(6)
    lens = lambda graph: {n: float(n) for n in graph.nodes()}

    recorded = {}
    monkeypatch.setattr(
        mapper, "update_metric", lambda name, value: recorded.update({name: value})
    )

    _, _, ov = mapper.tune_overlap(
        g, overlaps=[0.2, 0.3, 0.4, 0.5], n_intervals=2, lens=lens
    )
    assert recorded["mapper_overlap_opt"] == ov


def test_mapper_to_json_autotunes_by_default(monkeypatch):
    """`mapper_to_json` performs overlap tuning unless disabled."""

    g = nx.path_graph(6)
    called: dict[str, bool] = {}

    def fake_tune(*args, **kwargs):
        called["ran"] = True
        return nx.Graph(), [], 0.3

    monkeypatch.setattr(mapper, "tune_overlap", fake_tune)
    mapper.mapper_to_json(g)
    assert called.get("ran")


def test_cache_mapper_nerve_with_redis_and_lmdb(tmp_path):
    """Ensure caching uses Redis and LMDB layers."""
    g = nx.path_graph(4)
    client = fakeredis.FakeRedis()
    db = str(tmp_path / "cache.mdb")
    ssd = str(tmp_path / "ssd")
    nerve1, cover1 = mapper.cache_mapper_nerve(
        g, 1, redis_client=client, lmdb_path=db, ssd_dir=ssd
    )
    # second call should hit the cache
    nerve2, cover2 = mapper.cache_mapper_nerve(
        g, 1, redis_client=client, lmdb_path=db, ssd_dir=ssd
    )
    assert nx.is_isomorphic(nerve1, nerve2)
    assert cover1 == cover2
    assert client.get(f"1_{mapper._hash_graph(g)}") is not None

    # force LMDB fallback then SSD fallback
    client.flushall()
    nerve3, cover3 = mapper.cache_mapper_nerve(
        g, 1, redis_client=client, lmdb_path=db, ssd_dir=ssd
    )
    assert nx.is_isomorphic(nerve1, nerve3)
    assert cover1 == cover3
    # remove LMDB entry to trigger SSD read
    env = lmdb.open(db, readonly=False)
    env.close()
    import shutil

    shutil.rmtree(db)
    client.flushall()
    nerve4, _ = mapper.cache_mapper_nerve(
        g, 1, redis_client=client, lmdb_path=db, ssd_dir=ssd
    )
    assert nx.is_isomorphic(nerve1, nerve4)


def test_adjust_ttl_updates(monkeypatch):
    """TTL decreases when hit ratio is low."""

    monkeypatch.setattr(
        mapper, "os", type("O", (), {"getloadavg": lambda: (0, 0, 0)})()
    )
    start = mapper._redis_ttl = 1000
    mapper._redis_hits = 1
    mapper._redis_misses = 9
    mapper._hit_ema = 0
    mapper._last_ttl_eval = 0
    monkeypatch.setattr(mapper.time, "time", lambda: 301.0)
    mapper._adjust_ttl(None)
    assert mapper._redis_ttl < start


def test_adjust_ttl_high_load(monkeypatch):
    """High hit ratio combined with CPU load increases TTL."""
    monkeypatch.setattr(
        mapper,
        "os",
        type("O", (), {"getloadavg": lambda: (1.0,), "cpu_count": lambda: 1})(),
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


def test_l2_evict_and_delete(tmp_path):
    """Old LMDB entries should be purged and deletions logged."""

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


def test_eviction_thread_lifecycle(monkeypatch, tmp_path):
    """Ensure the eviction worker thread starts and stops."""

    def fake_config():
        return {"cache": {"l2_max_size_mb": 1, "l2_ttl_hours": 1}}

    monkeypatch.setattr(mapper, "load_config", fake_config)
    monkeypatch.setattr(mapper, "_evict_worker", lambda *a, **k: None)
    path = tmp_path / "lmdb.mdb"
    mapper.start_l2_eviction_thread(lmdb_path=str(path), interval=0.1)
    assert mapper._evict_thread is not None
    mapper.stop_l2_eviction_thread()
    assert mapper._evict_thread is None
