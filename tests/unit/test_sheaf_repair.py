import json
import time

import networkx as nx

import cron.sheaf_repair as repair


class DummyRedis:
    """Minimal Redis-like store used for tests."""

    def __init__(self):
        self.store: list[str] = []

    def delete(self, key: str) -> None:  # pragma: no cover - trivial
        self.store.clear()

    def rpush(self, key: str, value: str) -> None:
        self.store.append(value)

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        return self.store[start : end + 1]


class DummyDriver:
    def close(self):
        pass


class DummyDS:
    def __init__(self, g):
        self.graph = lambda: g


def test_collect_and_store_suggestions(monkeypatch):
    g = nx.Graph()
    for i in range(6):  # create edges 0-1, 1-2, ..., 5-6
        g.add_edge(str(i), str(i + 1))

    monkeypatch.setattr(
        repair,
        "top_k_incoherent",
        lambda g, top, tau: [((str(i), str(i + 1)), 1.0) for i in range(5)],
    )
    monkeypatch.setattr(repair, "sheaf_consistency_score", lambda g: 0.9)

    redis = DummyRedis()
    monkeypatch.setattr(repair, "get_redis_client", lambda: redis)
    monkeypatch.setattr(repair, "get_neo4j_driver", lambda: DummyDriver())
    DummyBuilder = type(
        "B", (), {"from_redis": classmethod(lambda cls, c, n, d: DummyDS(g))}
    )
    monkeypatch.setattr(repair, "DatasetBuilder", DummyBuilder)

    start = time.perf_counter()
    suggestions = repair.main("demo", top=5, tau=0.1)
    elapsed = time.perf_counter() - start
    assert len(suggestions) == 5
    assert elapsed < 0.2
    assert len(redis.store) == 5
    first = json.loads(redis.store[0])
    assert first["delete"].startswith("MATCH")
