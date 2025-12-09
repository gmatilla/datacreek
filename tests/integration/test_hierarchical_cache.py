import tempfile

import networkx as nx
import pytest

from datacreek.analysis.mapper import _cache_get, _cache_put, cache_mapper_nerve


def test_hierarchical_cache_roundtrip(tmp_path):
    fakeredis = pytest.importorskip("fakeredis")
    client = fakeredis.FakeRedis()
    g = nx.path_graph(3)
    cover = [{0, 1}, {1, 2}]
    lmdb_dir = tmp_path / "lmdb"
    ssd_dir = tmp_path / "ssd"
    _cache_put(
        "g1",
        g,
        cover,
        redis_client=client,
        lmdb_path=str(lmdb_dir),
        ssd_dir=str(ssd_dir),
        ttl=10,
    )
    res = _cache_get(
        "g1", redis_client=client, lmdb_path=str(lmdb_dir), ssd_dir=str(ssd_dir)
    )
    assert res is not None
    nerve, c = res
    assert nerve.number_of_edges() == g.number_of_edges()
    assert c == [set(x) for x in cover]


def test_cache_mapper_nerve_build(tmp_path):
    fakeredis = pytest.importorskip("fakeredis")
    client = fakeredis.FakeRedis()
    g = nx.path_graph(4)
    lmdb_dir = tmp_path / "lmdb"
    ssd_dir = tmp_path / "ssd"
    nerve, cover = cache_mapper_nerve(
        g,
        1,
        redis_client=client,
        lmdb_path=str(lmdb_dir),
        ssd_dir=str(ssd_dir),
        ttl=10,
    )
    assert nerve.number_of_nodes() == len(cover)
    # ensure value now cached
    nerve2, cover2 = cache_mapper_nerve(
        g,
        1,
        redis_client=client,
        lmdb_path=str(lmdb_dir),
        ssd_dir=str(ssd_dir),
        ttl=10,
    )
    assert nx.is_isomorphic(nerve2, nerve)
    assert cover2 == cover


def test_cache_put_ttl(tmp_path):
    fakeredis = pytest.importorskip("fakeredis")
    client = fakeredis.FakeRedis()
    g = nx.path_graph(3)
    cover = [{0, 1}, {1, 2}]
    lmdb_dir = tmp_path / "lmdb"
    ssd_dir = tmp_path / "ssd"
    _cache_put(
        "ttl",
        g,
        cover,
        redis_client=client,
        lmdb_path=str(lmdb_dir),
        ssd_dir=str(ssd_dir),
        ttl=5,
    )
    ttl_left = client.ttl("ttl")
    assert ttl_left is not None and ttl_left > 0
