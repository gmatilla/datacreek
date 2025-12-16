"""Integration tests for the public API."""

import json
import os
import time
from pathlib import Path

import fakeredis
from fastapi.testclient import TestClient

from datacreek.core.dataset import DatasetBuilder
from datacreek.models.stage import DatasetStage
from datacreek.models.task_status import TaskStatus
from datacreek.pipelines import DatasetType

os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")
os.environ["DATACREEK_REQUIRE_PERSISTENCE"] = "0"

from datacreek.api import app, get_db
from datacreek.db import Dataset, SessionLocal, SourceData, init_db

# Use a test database
TEST_DB_URL = "sqlite:///test_api.db"
os.environ["DATABASE_URL"] = TEST_DB_URL

# Re-initialize the DB for testing
init_db()

def get_test_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = get_test_db

client = TestClient(app)

def teardown_module(module):
    """Remove leftover test database."""
    app.dependency_overrides.clear()
    if os.path.exists("test_api.db"):
        os.remove("test_api.db")





def test_generate_params_validation() -> None:
    """Validate required generation parameters."""
    res = client.post(
        "/tasks/generate",
        json={"src_id": 1, "provider": ""},
    )
    if res.status_code != 422:
        raise AssertionError


def _wait_task(task_id: str) -> dict:
    for _ in range(50):
        res = client.get(f"/tasks/{task_id}")
        body = res.json()
        if body["status"] == "finished":
            return body["result"]
        time.sleep(0.01)
    raise AssertionError("task did not finish")


def test_async_pipeline(monkeypatch, tmp_path):
    redis_client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.api.get_neo4j_driver", lambda: None)

    def dummy_generate(path, *args, document_text=None, **kwargs):
        assert kwargs.get("config_overrides") == {"generation": {"temperature": 0.1}}
        assert kwargs.get("provider") == "api-endpoint"
        return {"qa_pairs": [{"question": "q", "answer": "a"}]}

    def dummy_curate(data, output_path=None, *args, **kwargs):
        return data

    def dummy_save(data, fmt, cfg, storage):
        return "converted"

    monkeypatch.setattr("datacreek.tasks.generate_data", dummy_generate)
    monkeypatch.setattr("datacreek.tasks.curate_qa_pairs", dummy_curate)
    monkeypatch.setattr("datacreek.tasks.convert_format", dummy_save)


    monkeypatch.setattr("datacreek.tasks.generate_data", dummy_generate)
    monkeypatch.setattr("datacreek.tasks.curate_qa_pairs", dummy_curate)
    monkeypatch.setattr("datacreek.tasks.convert_format", dummy_save)

    src_file = tmp_path / "src.txt"
    src_file.write_text("Paris is the capital of France.")

    res = client.post(
        "/tasks/ingest",
        json={"path": str(src_file), "extract_entities": True, "extract_facts": True},
    )
    task_id = res.json()["task_id"]
    result = _wait_task(task_id)
    src_id = result["id"]
    with SessionLocal() as db:
        src = db.get(SourceData, src_id)
        assert src.entities and src.facts

    res = client.post(
        "/tasks/generate",
        json={
            "src_id": src_id,
            "provider": "api-endpoint",
            "generation": {"temperature": 0.1},
        },
    )
    task_id = res.json()["task_id"]
    result = _wait_task(task_id)
    ds_id = result["id"]

    res = client.post("/tasks/curate", json={"ds_id": ds_id})
    task_id = res.json()["task_id"]
    _wait_task(task_id)

    res = client.post(
        "/tasks/save", json={"ds_id": ds_id, "fmt": "jsonl"}
    )
    task_id = res.json()["task_id"]
    _wait_task(task_id)

    res = client.get("/datasets")
    assert len(res.json()) == 1

    # download dataset
    res = client.get(f"/datasets/{ds_id}/download")
    assert res.status_code == 200

    # update dataset path
    new_path = str(tmp_path / "new.json")
    res = client.patch(f"/datasets/{ds_id}", json={"path": new_path})
    assert res.json()["path"] == new_path

    res = client.delete(f"/datasets/{ds_id}")
    assert res.status_code in {200, 404}
    if res.status_code == 200:
        assert res.json()["status"] == "deleted"
        with SessionLocal() as db:
            ds = db.get(Dataset, ds_id)
            assert ds is None


def test_dataset_history_route(monkeypatch):
    redis_client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = redis_client
    ds.add_document("d1", source="s")
    ds.to_redis(redis_client, "dataset:demo")
    redis_client.sadd("datasets", "demo")

    res = client.get("/datasets/demo/history")
    assert res.status_code == 200
    data = res.json()
    assert any(ev["operation"] == "add_document" for ev in data)


def test_global_event_log_route(monkeypatch):
    redis_client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.backends.get_redis_client", lambda config_path=None: redis_client)
    ds = DatasetBuilder(DatasetType.TEXT, name="glob")
    ds.redis_client = redis_client
    ds.add_document("d1", source="s")
    ds.add_chunk("d1", "c1", "hi")
    ds.to_redis(redis_client, "dataset:glob")
    redis_client.sadd("datasets", "glob")

    res = client.get("/datasets/events")
    assert res.status_code == 200
    data = res.json()
    assert any(
        ev["dataset"] == "glob" and ev["operation"] == "add_chunk" for ev in data
    )


def test_dataset_progress_route(monkeypatch):
    redis_client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = redis_client
    ds.to_redis(redis_client, "dataset:demo")
    redis_client.hset(
        "dataset:demo:progress",
        mapping={"status": TaskStatus.INGESTING.value, "count": "1"},
    )

    res = client.get("/datasets/demo/progress")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == TaskStatus.INGESTING.value
    assert data["count"] == 1


def test_dataset_progress_history_route(monkeypatch):
    redis_client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = redis_client
    ds.to_redis(redis_client, "dataset:demo")
    redis_client.rpush(
        "dataset:demo:progress:history",
        json.dumps({"status": TaskStatus.INGESTING.value}),
    )
    redis_client.rpush(
        "dataset:demo:progress:history",
        json.dumps({"status": TaskStatus.COMPLETED.value}),
    )

    res = client.get("/datasets/demo/progress/history")
    assert res.status_code == 200
    data = res.json()
    assert data[0]["status"] == TaskStatus.INGESTING.value
    assert data[-1]["status"] == TaskStatus.COMPLETED.value





def test_dataset_export_route(monkeypatch):
    redis_client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = redis_client
    ds.to_redis(redis_client, "dataset:demo")
    redis_client.set("dataset:demo:export:jsonl", '{"hello": 1}')

    res = client.get("/datasets/demo/export")
    assert res.status_code == 200
    assert res.json() == {"hello": 1}


def test_dataset_export_route_uses_progress_key(monkeypatch):
    redis_client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = redis_client
    ds.to_redis(redis_client, "dataset:demo")
    redis_client.set("custom:key", '{"v": 2}')
    redis_client.hset(
        "dataset:demo:progress",
        "export",
        json.dumps({"fmt": "jsonl", "key": "custom:key"}),
    )

    res = client.get("/datasets/demo/export")
    assert res.status_code == 200
    assert res.json() == {"v": 2}





def test_graph_progress_route(monkeypatch):
    redis_client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = redis_client
    ds.to_redis(redis_client, "graph:demo")
    redis_client.hset(
        "graph:demo:progress", mapping={"status": "saving_neo4j", "progress": "0.5"}
    )

    res = client.get("/graphs/demo/progress")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "saving_neo4j"
    assert float(data["progress"]) == 0.5


def test_graph_progress_history_route(monkeypatch):
    redis_client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = redis_client
    ds.to_redis(redis_client, "graph:demo")
    redis_client.rpush(
        "graph:demo:progress:history",
        json.dumps({"status": TaskStatus.SAVING_NEO4J.value}),
    )
    redis_client.rpush(
        "graph:demo:progress:history",
        json.dumps({"status": TaskStatus.COMPLETED.value}),
    )

    res = client.get("/graphs/demo/progress/history")
    assert res.status_code == 200
    data = res.json()
    assert data[0]["status"] == TaskStatus.SAVING_NEO4J.value
    assert data[-1]["status"] == TaskStatus.COMPLETED.value





def test_dataset_versions_route(monkeypatch):
    redis_client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = redis_client
    ds.versions.append({"params": {}, "time": "t", "result": {"v": 1}})
    ds.to_redis(redis_client, "dataset:demo")
    redis_client.sadd("datasets", "demo")

    res = client.get("/datasets/demo/versions")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert data and data[0]["index"] == 1
    assert data[0]["result"]["v"] == 1


def test_dataset_version_item_route(monkeypatch):
    redis_client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = redis_client
    ds.versions.extend(
        [
            {"params": {}, "time": "t1", "result": {"v": 1}},
            {"params": {}, "time": "t2", "result": {"v": 2}},
        ]
    )
    ds.to_redis(redis_client, "dataset:demo")
    redis_client.sadd("datasets", "demo")

    res = client.get("/datasets/demo/versions/2")
    assert res.status_code == 200
    data = res.json()
    assert data["result"]["v"] == 2


def test_dataset_version_delete_route(monkeypatch):
    redis_client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = redis_client
    ds.versions.extend(
        [
            {"params": {}, "time": "t1", "result": {"v": 1}},
            {"params": {}, "time": "t2", "result": {"v": 2}},
        ]
    )
    ds.to_redis(redis_client, "dataset:demo")
    redis_client.sadd("datasets", "demo")

    res = client.delete("/datasets/demo/versions/1")
    assert res.status_code == 200
    assert redis_client.exists("dataset:demo")
    ds_loaded = DatasetBuilder.from_redis(redis_client, "dataset:demo")
    assert len(ds_loaded.versions) == 1 and ds_loaded.versions[0]["result"]["v"] == 2





def test_dataset_route_bad_name(monkeypatch):
    redis_client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)

    res = client.get("/datasets/bad name/history")
    assert res.status_code == 422

    res = client.get("/datasets/bad name/progress/history")
    assert res.status_code == 422


def test_create_dataset_route(monkeypatch):
    redis_client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)

    res = client.post("/datasets/newds", json={"dataset_type": "qa"})
    assert res.status_code == 201
    assert redis_client.exists("dataset:newds")


def test_create_dataset_route_bad_name(monkeypatch):
    redis_client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)

    res = client.post(
        "/datasets/bad name", json={"dataset_type": "qa"}
    )
    assert res.status_code == 422
    long_name = "a" * 65
    res = client.post(
        f"/datasets/{long_name}", json={"dataset_type": "qa"}
    )
    assert res.status_code == 422


def test_create_dataset_route_exists(monkeypatch):
    redis_client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)

    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = redis_client
    ds.to_redis(redis_client, "dataset:demo")
    redis_client.sadd("datasets", "demo")

    res = client.post("/datasets/demo", json={"dataset_type": "text"})
    assert res.status_code == 409


def test_list_user_datasets(monkeypatch):
    redis_client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    redis_client.sadd("datasets", "demo")

    res = client.get("/users/me/datasets")
    assert res.status_code == 200
    assert "demo" in res.json()


def test_list_user_datasets_details(monkeypatch):
    redis_client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    redis_client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = redis_client
    ds.stage = DatasetStage.GENERATED
    ds.to_redis(redis_client, "dataset:demo")
    redis_client.sadd("datasets", "demo")
    redis_client.hset(
        "dataset:demo:progress", mapping={"status": TaskStatus.INGESTING.value}
    )

    res = client.get("/users/me/datasets/details")
    assert res.status_code == 200
    data = res.json()
    assert data[0]["name"] == "demo"
    assert data[0]["stage"] == DatasetStage.GENERATED
    assert data[0]["progress"]["status"] == TaskStatus.INGESTING.value


def test_delete_persisted_dataset_route(monkeypatch):
    redis_client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.backends.get_redis_client", lambda config_path=None: redis_client)
    monkeypatch.setattr("datacreek.api.get_neo4j_driver", lambda: None)

    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = redis_client
    ds.to_redis(redis_client, "dataset:demo")
    redis_client.sadd("datasets", "demo")

    res = client.delete("/datasets/demo")
    assert res.status_code == 202
    assert not redis_client.exists("dataset:demo")
    assert b"demo" not in redis_client.smembers("datasets")





def test_explain_endpoint(monkeypatch):
    redis_client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.backends.get_redis_client", lambda config_path=None: redis_client)
    monkeypatch.setattr("datacreek.api.get_neo4j_driver", lambda: None)

    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = redis_client
    ds.add_document("d1", source="s")
    ds.add_chunk("d1", "c1", "a")
    ds.add_chunk("d1", "c2", "b")
    ds.graph.graph.add_edge("c1", "c2", relation="sim")
    ds.graph.graph.nodes["c1"]["embedding"] = [0.0, 0.0]
    ds.graph.graph.nodes["c2"]["embedding"] = [1.0, 0.0]
    ds.to_redis(redis_client, "dataset:demo")
    redis_client.sadd("datasets", "demo")

    res = client.get("/datasets/demo/explain/c1")
    assert res.status_code == 200
    body = res.json()
    assert "nodes" in body and "edges" in body and "attention" in body

    res = client.get("/datasets/demo/explain/c1?fmt=svg", headers=headers)
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("image/svg")

    import importlib.util

    if importlib.util.find_spec("cairosvg"):
        res = client.get("/datasets/demo/explain/c1?fmt=png")
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("image/png")
    else:
        res = client.get("/datasets/demo/explain/c1?fmt=png")
        assert res.status_code == 501


import pytest


@pytest.mark.skip("DB dependencies not available in CI")
def test_vector_search_endpoint(monkeypatch):
    redis_client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.api.get_neo4j_driver", lambda: None)

    user = type("U", (), {"id": 1})()
    import importlib

    vector_router_mod = importlib.import_module("datacreek.routers.vector_router")

    monkeypatch.setattr("datacreek.api.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.services.get_redis_client", lambda: redis_client)
    monkeypatch.setattr("datacreek.api.get_neo4j_driver", lambda: None)

    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = redis_client
    ds.add_document("d1", source="s")
    ds.add_chunk("d1", "c1", "hello world")
    ds.add_chunk("d1", "c2", "another text")
    ds.to_redis(redis_client, "dataset:demo")
    redis_client.sadd("datasets", "demo")

    res = client.post(
        "/vector/search",
        json={"dataset": "demo", "query": "hello"},
    )
    assert res.status_code == 200
    assert res.json()[0] == "c1"
