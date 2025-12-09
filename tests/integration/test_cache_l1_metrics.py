import networkx as nx
import pytest

from datacreek.analysis.mapper import _cache_get, _cache_put


def test_cache_l1_metrics(tmp_path):
    fakeredis = pytest.importorskip("fakeredis")
    client = fakeredis.FakeRedis()
    g = nx.path_graph(3)
    cover = [{0, 1}, {1, 2}]
    lmdb_dir = tmp_path / "lmdb"
    ssd_dir = tmp_path / "ssd"
    _cache_put(
        "g",
        g,
        cover,
        redis_client=client,
        lmdb_path=str(lmdb_dir),
        ssd_dir=str(ssd_dir),
        ttl=10,
    )
    assert client.get("hits") is None
    assert int(client.get("miss")) == 1
    res = _cache_get(
        "g",
        redis_client=client,
        lmdb_path=str(lmdb_dir),
        ssd_dir=str(ssd_dir),
    )
    assert res is not None
    assert int(client.get("hits")) == 1
    missing = _cache_get(
        "missing",
        redis_client=client,
        lmdb_path=str(lmdb_dir),
        ssd_dir=str(ssd_dir),
    )
    assert missing is None
    _cache_put(
        "missing",
        g,
        cover,
        redis_client=client,
        lmdb_path=str(lmdb_dir),
        ssd_dir=str(ssd_dir),
        ttl=10,
    )
    assert int(client.get("miss")) == 2


def test_cache_l1_ttl_adjustment(monkeypatch):
    """TTL adapts to hit ratio and updates Prometheus gauge."""

    fakeredis = pytest.importorskip("fakeredis")
    client = fakeredis.FakeRedis()

    import datacreek.analysis.mapper as mapper

    mapper._redis_ttl = 3600
    mapper._redis_hits = 9
    mapper._redis_misses = 1
    mapper._last_ttl_eval = 0
    mapper._hit_ema = 0.0

    gauge_val = {}

    class DummyGauge:
        def set(self, value):
            gauge_val["ratio"] = value

    monkeypatch.setattr("datacreek.analysis.monitoring.redis_hit_ratio", DummyGauge())
    monkeypatch.setattr(mapper.time, "time", lambda: 301)

    mapper._adjust_ttl(client, None)
    assert mapper._redis_ttl > 3600
    assert gauge_val["ratio"] == pytest.approx(0.9)

    mapper._redis_ttl = 3600
    mapper._redis_hits = 0
    mapper._redis_misses = 10
    mapper._last_ttl_eval = 0
    mapper._hit_ema = 0.0
    monkeypatch.setattr(mapper.time, "time", lambda: 302)

    mapper._adjust_ttl(client, None)
    assert mapper._redis_ttl < 3600


def test_cpu_load_adapts_ttl(monkeypatch):
    """TTL increases to avoid thrash when CPU load is high."""

    fakeredis = pytest.importorskip("fakeredis")
    client = fakeredis.FakeRedis()

    import datacreek.analysis.mapper as mapper

    mapper._redis_ttl = 600
    mapper._redis_hits = 10
    mapper._redis_misses = 0
    mapper._last_ttl_eval = 0

    monkeypatch.setattr(mapper.os, "getloadavg", lambda: (8.0, 0.0, 0.0))
    monkeypatch.setattr(mapper.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(mapper.time, "time", lambda: 301)

    mapper._adjust_ttl(client, "foo")
    assert mapper._redis_ttl >= mapper._ttl_max
