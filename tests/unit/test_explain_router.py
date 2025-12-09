import base64
import importlib
import json
import time
import types

import networkx as nx
import pytest
from fastapi import HTTPException

# Import the module itself rather than the router object aggregated in
# ``datacreek.routers`` so we can monkeypatch its attributes.
router = importlib.import_module("datacreek.routers.explain_router")


class DummyUser:
    def __init__(self, id):
        self.id = id


class DummyDS:
    def __init__(self, owner_id=1):
        self.owner_id = owner_id
        self.redis_client = None

    def explain_node(self, node, hops=3):
        return {"nodes": [node], "edges": [], "attention": {node: 1.0}}

    def _record_event(self, *a, **k):
        pass


class DummyDriver:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class DummyRedis:
    def __init__(self):
        self.closed = False

    def hset(self, *a, **k):
        pass


def test_get_current_user(monkeypatch):
    user = DummyUser(3)
    monkeypatch.setattr(router, "get_user_by_key", lambda db, key: user)
    out = router.get_current_user(api_key="token")
    assert out is user
    monkeypatch.setattr(router, "get_user_by_key", lambda db, key: None)
    with pytest.raises(HTTPException):
        router.get_current_user(api_key="bad")


def test_load_dataset_success(monkeypatch):
    dummy = DummyDS(owner_id=3)
    driver = DummyDriver()
    monkeypatch.setattr(router, "get_redis_client", lambda: DummyRedis())
    monkeypatch.setattr(router, "get_neo4j_driver", lambda: driver)
    DummyBuilder = type(
        "B", (), {"from_redis": classmethod(lambda cls, c, n, d: dummy)}
    )
    monkeypatch.setattr(router, "DatasetBuilder", DummyBuilder)
    user = DummyUser(3)
    ds = router._load_dataset("demo", user)
    assert ds is dummy
    assert dummy.redis_client is not None
    assert driver.closed


def test_load_dataset_errors(monkeypatch):
    monkeypatch.setattr(router, "get_redis_client", lambda: None)
    with pytest.raises(HTTPException):
        router._load_dataset("x", DummyUser(1))
    monkeypatch.setattr(router, "get_redis_client", lambda: DummyRedis())
    monkeypatch.setattr(router, "get_neo4j_driver", lambda: DummyDriver())
    DummyBuilder = type(
        "B",
        (),
        {
            "from_redis": classmethod(
                lambda cls, c, n, d: (_ for _ in ()).throw(KeyError())
            )
        },
    )
    monkeypatch.setattr(router, "DatasetBuilder", DummyBuilder)
    with pytest.raises(HTTPException):
        router._load_dataset("x", DummyUser(1))
    dummy = DummyDS(owner_id=2)
    monkeypatch.setattr(router.DatasetBuilder, "from_redis", lambda c, n, d: dummy)
    with pytest.raises(HTTPException):
        router._load_dataset("x", DummyUser(1))


def test_explain_node_public(monkeypatch):
    dummy = DummyDS(owner_id=1)
    monkeypatch.setattr(router, "_load_dataset", lambda name, user: dummy)
    monkeypatch.setattr(router, "explain_to_svg", lambda data: "<svg></svg>")
    user = DummyUser(1)
    resp = router.explain_node_public("foo", user=user)
    payload = resp.body.decode()
    data = json.loads(payload)
    assert data["nodes"] == ["foo"]
    assert base64.b64decode(data["svg"]).decode() == "<svg></svg>"


def test_sheaf_diff(monkeypatch):
    dummy = DummyDS(owner_id=1)
    g = nx.Graph()
    g.add_edge(0, 1, sheaf_sign=1)
    dummy.graph = lambda: g
    monkeypatch.setattr(router, "_load_dataset", lambda name, user: dummy)
    monkeypatch.setattr(router, "top_k_incoherent", lambda g, top, tau: [((0, 1), 0.5)])
    user = DummyUser(1)
    resp = router.sheaf_diff(user=user)
    data = json.loads(resp.body.decode())
    assert data["edges"][0]["u"] == 0
    assert data["edges"][0]["delta"] == 0.5


def test_sheaf_diff_latency(monkeypatch):
    dummy = DummyDS(owner_id=1)
    g = nx.Graph()
    g.add_edge(0, 1, sheaf_sign=1)
    dummy.graph = lambda: g
    monkeypatch.setattr(router, "_load_dataset", lambda name, user: dummy)
    monkeypatch.setattr(
        router,
        "top_k_incoherent",
        lambda g, top, tau: [((0, 1), 0.5)] * top,
    )
    user = DummyUser(1)
    start = time.perf_counter()
    router.sheaf_diff(user=user, top=50)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.1


def test_repair_preview(monkeypatch):
    redis = types.SimpleNamespace(
        lrange=lambda k, s, e: [
            json.dumps({"u": "a", "v": "b", "delta": 1.0, "delete": "D", "merge": "M"})
        ]
    )
    monkeypatch.setattr(router, "get_redis_client", lambda: redis)
    user = DummyUser(1)
    resp = router.repair_preview(user=user, limit=1)
    data = json.loads(resp.body.decode())
    assert data["suggestions"][0]["u"] == "a"
