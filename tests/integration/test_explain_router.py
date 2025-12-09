import os
import sys

import fakeredis
from fastapi.testclient import TestClient

os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")
os.environ.setdefault("DATABASE_URL", "sqlite:///test_explain.db")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import datacreek.api
from datacreek.api import app
from datacreek.core.dataset import DatasetBuilder
from datacreek.pipelines import DatasetType

client = TestClient(app)


def test_explain_router(monkeypatch):
    redis_client = fakeredis.FakeStrictRedis()
    import importlib

    exr = importlib.import_module("datacreek.routers.explain_router")
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    monkeypatch.setattr(exr, "get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.api.get_neo4j_driver", lambda: None)
    monkeypatch.setattr(exr, "get_neo4j_driver", lambda: None)

    from datacreek.db import User

    app.dependency_overrides[exr.get_current_user] = lambda: User(
        id=1, username="bob", api_key="k"
    )
    app.dependency_overrides[datacreek.api.get_current_user] = lambda: User(
        id=1, username="bob", api_key="k"
    )
    key = "k"
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    monkeypatch.setattr(ds, "_enforce_policy", lambda *a, **k: None)
    ds.owner_id = 1
    ds.redis_client = redis_client
    ds.add_document("d1", source="s")
    ds.add_chunk("d1", "c1", "a")
    ds.add_chunk("d1", "c2", "b")
    ds.graph.graph.add_edge("c1", "c2", relation="sim")
    ds.graph.graph.nodes["c1"]["embedding"] = [0.0, 0.0]
    ds.graph.graph.nodes["c2"]["embedding"] = [1.0, 0.0]
    ds.to_redis(redis_client, "dataset:demo")
    redis_client.sadd("datasets", "demo")
    monkeypatch.setattr(exr, "_load_dataset", lambda name, user: ds)

    headers = {"X-API-Key": key}
    url = "/explain/c1?dataset=demo"
    res = client.get(url, headers=headers)
    assert res.status_code == 200
    body = res.json()
    assert body["nodes"] and body["edges"] and body["attn"] is not None
    assert "svg" in body and body["svg"].startswith("PD94") is False
