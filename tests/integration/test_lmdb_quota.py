import logging
import pickle

import networkx as nx
import pytest

from datacreek.analysis.mapper import _cache_put


class FakeEnv:
    def __init__(self):
        self._closed = False

    def set_mapsize(self, size):
        self.map_size = size

    def stat(self):
        return {"psize": 4096}

    def info(self):
        return {"map_size": int(0.95 * 1024 * 1024)}

    def begin(self, write=False):
        class Txn:
            def __enter__(self2):
                return self2

            def __exit__(self2, exc_type, exc, tb):
                pass

            def put(self2, key, val):
                pass

            def cursor(self2):
                class C:
                    def first(self):
                        return False

                return C()

        return Txn()

    def sync(self):
        pass

    def close(self):
        self._closed = True


def test_lmdb_soft_quota(monkeypatch, tmp_path):
    fakeredis = pytest.importorskip("fakeredis")
    client = fakeredis.FakeRedis()
    g = nx.path_graph(3)
    cover = [{0, 1}, {1, 2}]

    logged = {"warn": 0}

    def warn(msg, *args):
        logged["warn"] += 1

    monkeypatch.setattr(logging.getLogger("datacreek.analysis.mapper"), "warning", warn)

    monkeypatch.setattr(
        "datacreek.analysis.mapper.lmdb",
        type("L", (), {"open": lambda *a, **k: FakeEnv()}),
    )
    monkeypatch.setattr(
        "datacreek.analysis.mapper.load_config",
        lambda: {"cache": {"l2_max_size_mb": 1, "l2_ttl_hours": 1}},
    )

    _cache_put(
        "x",
        g,
        cover,
        redis_client=client,
        lmdb_path=str(tmp_path / "lmdb"),
        ssd_dir=str(tmp_path / "ssd"),
        ttl=1,
    )
    assert logged["warn"] == 1
