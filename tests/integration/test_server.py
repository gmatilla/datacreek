import importlib
import json
import os
import sys

import fakeredis
import requests
from werkzeug.security import generate_password_hash

os.environ["DATABASE_URL"] = "sqlite:///test_server.db"
if os.path.exists("test_server.db"):
    os.remove("test_server.db")
if "datacreek.db" in sys.modules:
    importlib.reload(sys.modules["datacreek.db"])
import datacreek.db as db

db.init_db()
with db.SessionLocal() as session:
    user = db.User(
        username="alice",
        api_key="key",
        password_hash=generate_password_hash("pw"),
    )
    session.add(user)
    session.commit()

import datacreek.server.app as app_module
from datacreek.core.dataset import DatasetBuilder
from datacreek.core.knowledge_graph import KnowledgeGraph
from datacreek.db import verify_password
from datacreek.models.export_format import ExportFormat
from datacreek.pipelines import DatasetType
from datacreek.server.app import DATASETS, app
from datacreek.tasks import dataset_export_task

app.config["WTF_CSRF_ENABLED"] = False


def _login(client):
    return client.post(
        "/api/login",
        json={"username": "alice", "password": "pw"},
    )


def test_register_and_login():
    with app.test_client() as client:
        res = client.post(
            "/api/register",
            json={"username": "bob", "password": "pw"},
        )
        data = res.get_json()
        assert "api_key" in data
        # Now login with new user
        res = client.post(
            "/api/login",
            json={"username": "bob", "password": "pw"},
        )
        assert res.status_code == 200


def test_login_required_redirect():
    with app.test_client() as client:
        res = client.get("/datasets")
        assert res.status_code == 401


def test_dataset_graph_route():
    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.add_document("d1", source="s")
    ds.add_chunk("d1", "c1", "text")
    DATASETS["demo"] = ds

    with app.test_client() as client:
        _login(client)
        res = client.get("/datasets/demo/graph")
        assert res.status_code == 200
        data = res.get_json()
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1
    DATASETS.clear()


def test_dataset_search_route():
    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.add_document("d1", source="s")
    ds.add_chunk("d1", "c1", "hello world")
    DATASETS["demo"] = ds

    with app.test_client() as client:
        _login(client)
        res = client.get("/datasets/demo/search", query_string={"q": "hello"})
        assert res.status_code == 200
        data = res.get_json()
        assert data == ["c1"]
    DATASETS.clear()


def test_dataset_ingest_route(tmp_path):
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    DATASETS["demo"] = ds

    f = tmp_path / "doc.txt"
    f.write_text("hello world")

    with app.test_client() as client:
        _login(client)
        res = client.post(
            "/datasets/demo/ingest",
            data={"input_path": str(f), "doc_id": "doc1"},
            follow_redirects=True,
        )
        assert res.status_code == 200
    assert ds.search_chunks("hello") == ["doc1_chunk_0"]
    DATASETS.clear()


def test_api_dataset_ingest_resolves_path(tmp_path, monkeypatch):
    client = fakeredis.FakeStrictRedis()
    app_module.REDIS = client
    monkeypatch.setattr("datacreek.tasks.get_redis_client", lambda: client)
    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: None)
    import datacreek.tasks as tasks_mod

    tasks_mod.celery_app.conf.task_always_eager = True

    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.to_redis(client, "dataset:demo")
    DATASETS["demo"] = ds

    f = tmp_path / "doc.txt"
    f.write_text("hello world")

    with app.test_client() as cl:
        _login(cl)
        res = cl.post(
            "/api/datasets/demo/ingest",
            json={"path": str(f), "high_res": True, "extract_entities": True},
        )
        assert res.status_code == 200

        res = cl.get("/api/datasets/demo/progress")
        data = res.get_json()
        assert data.get("ingested") == 1
        assert data.get("last_ingested", {}).get("path") == str(f)
        assert "time" in data.get("last_ingested", {})
        assert data.get("ingest_start") is not None
        assert data.get("ingest_finish") is not None
        params = data.get("ingestion_params")
        assert params.get("high_res") is True
        assert params.get("extract_entities") is True

    loaded = DatasetBuilder.from_redis(client, "dataset:demo")
    assert loaded.search_chunks("hello") == ["doc_chunk_0"]
    DATASETS.clear()


def test_dataset_detail_route():
    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.add_document("d1", source="s")
    ds.add_chunk("d1", "c1", "hello")
    DATASETS["demo"] = ds

    with app.test_client() as client:
        _login(client)
        res = client.get("/api/datasets/demo")
        assert res.status_code == 200
        data = res.get_json()
        assert data["id"] == ds.id
        assert data["size"] == len("hello")
    DATASETS.clear()


def test_save_dataset_neo4j(monkeypatch):
    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.add_document("d1", source="s")
    DATASETS["demo"] = ds

    called = {}

    def fake_to_neo4j(self, driver, clear=True, dataset=None):
        called["called"] = True

    monkeypatch.setattr(KnowledgeGraph, "to_neo4j", fake_to_neo4j)

    class DummyDriver:
        def close(self):
            pass

    monkeypatch.setattr(app_module, "get_neo4j_driver", lambda: DummyDriver())

    with app.test_client() as client:
        _login(client)
        res = client.post("/datasets/demo/save_neo4j")
        assert res.status_code == 302
    assert called.get("called")
    DATASETS.clear()


def test_load_dataset_neo4j(monkeypatch):
    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.add_document("d1", source="s")
    DATASETS["demo"] = ds

    new_graph = KnowledgeGraph()
    new_graph.add_document("n", source="a")
    monkeypatch.setattr(
        KnowledgeGraph,
        "from_neo4j",
        staticmethod(lambda driver, dataset=None: new_graph),
    )

    class DummyDriver:
        def close(self):
            pass

    monkeypatch.setattr(app_module, "get_neo4j_driver", lambda: DummyDriver())

    with app.test_client() as client:
        _login(client)
        res = client.post("/datasets/demo/load_neo4j")
        assert res.status_code == 302
    assert ds.graph is new_graph
    DATASETS.clear()


def test_api_neo4j_endpoints(tmp_path, monkeypatch):
    client = fakeredis.FakeStrictRedis()
    app_module.REDIS = client
    monkeypatch.setattr("datacreek.tasks.get_redis_client", lambda: client)
    import datacreek.tasks as tasks_mod

    tasks_mod.celery_app.conf.task_always_eager = True

    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.add_document("d", source="s")
    ds.to_redis(client, "dataset:demo")
    DATASETS["demo"] = ds

    class DummyDriver:
        def close(self):
            pass

    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: DummyDriver())

    new_graph = KnowledgeGraph()
    new_graph.add_document("n", source="a")
    monkeypatch.setattr(ds.graph.__class__, "to_neo4j", lambda *a, **k: None)
    monkeypatch.setattr(
        ds.graph.__class__,
        "from_neo4j",
        staticmethod(lambda driver, dataset=None: new_graph),
    )

    with app.test_client() as cl:
        _login(cl)
        res = cl.post("/api/datasets/demo/save_neo4j")
        assert res.status_code == 200
        prog = cl.get("/api/datasets/demo/progress").get_json()
        assert prog.get("save_neo4j") is not None
        assert "time" in prog.get("save_neo4j")
        assert prog.get("save_neo4j_start") is not None
        assert prog.get("save_neo4j_finish") is not None
        res = cl.post("/api/datasets/demo/load_neo4j")
        assert res.status_code == 200
        prog = cl.get("/api/datasets/demo/progress").get_json()
        assert prog.get("load_neo4j") is not None
        assert "time" in prog.get("load_neo4j")
        assert prog.get("load_neo4j_start") is not None
        assert prog.get("load_neo4j_finish") is not None

    loaded = DatasetBuilder.from_redis(client, "dataset:demo")
    assert loaded.graph.search_documents("n") == ["n"]
    DATASETS.clear()


def test_api_search_endpoints():
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.add_document("d", source="s")
    ds.add_section("d", "s1", title="Intro")
    ds.add_section("d", "s2", title="Introduction")
    ds.add_section("d", "s3", title="Other")
    ds.add_chunk("d", "c1", "hello world")
    ds.add_chunk("d", "c2", "hello planet")
    ds.add_chunk("d", "c3", "other text")
    ds.graph.index.build()
    ds.link_similar_chunks(k=1)
    DATASETS["demo"] = ds

    with app.test_client() as client:
        _login(client)
        res = client.get(
            "/api/datasets/demo/search_hybrid", query_string={"q": "hello", "k": 1}
        )
        assert res.status_code == 200
        assert res.get_json()[0] == "c1"

        res = client.get(
            "/api/datasets/demo/search_links",
            query_string={"q": "hello", "k": 1, "hops": 1},
        )
        assert res.status_code == 200
        ids = [r["id"] for r in res.get_json()]
        assert "c1" in ids and "c2" in ids

        res = client.get(
            "/api/datasets/demo/similar_chunks",
            query_string={"cid": "c1", "k": 2},
        )
        assert res.status_code == 200
        sims = res.get_json()
        assert "c1" not in sims
        assert "c2" in sims

        res = client.get(
            "/api/datasets/demo/similar_chunks_data",
            query_string={"cid": "c1", "k": 2},
        )
        assert res.status_code == 200
        data = res.get_json()
        ids = [d["id"] for d in data]
        assert "c1" not in ids
        assert "c2" in ids

        res = client.get(
            "/api/datasets/demo/chunk_neighbors",
            query_string={"k": 1},
        )
        assert res.status_code == 200
        neighbors = res.get_json()
        assert neighbors["c1"][0] == "c2"

        res = client.get(
            "/api/datasets/demo/chunk_neighbors_data",
            query_string={"k": 1},
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["c1"][0]["id"] in {"c2", "c3"}

        res = client.get(
            "/api/datasets/demo/chunk_context",
            query_string={"cid": "c2", "before": 1, "after": 1},
        )
        assert res.status_code == 200
        assert res.get_json() == ["c1", "c2", "c3"]

        res = client.get(
            "/api/datasets/demo/similar_sections",
            query_string={"sid": "s1", "k": 2},
        )
        assert res.status_code == 200
        sims = res.get_json()
        assert "s1" not in sims
        assert "s2" in sims

        res = client.get(
            "/api/datasets/demo/similar_documents",
            query_string={"did": "d", "k": 1},
        )
        assert res.status_code == 200
        assert res.get_json() == []

        res = client.post(
            "/api/datasets/demo/section_similarity",
            query_string={"k": 1},
        )
        assert res.status_code == 200
        res = client.post(
            "/api/datasets/demo/document_similarity",
            query_string={"k": 1},
        )
        assert res.status_code == 200
        assert ("s1", "s2") in ds.graph.graph.edges
    DATASETS.clear()


def test_dataset_ops_endpoints(monkeypatch):
    client = fakeredis.FakeStrictRedis()
    app_module.REDIS = client
    monkeypatch.setattr("datacreek.tasks.get_redis_client", lambda: client)
    import datacreek.tasks as tasks_mod

    tasks_mod.celery_app.conf.task_always_eager = True

    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.redis_client = client
    ds.add_document("d1", source="s")
    ds.add_chunk("d1", "c1", "hello")
    ds.add_chunk("d1", "c2", "hello")
    ds.add_document("d2", source="bad")
    ds.add_chunk("d2", "c3", "bad text")
    ds.add_entity("e1", "Beethoven")
    ds.add_entity("e2", "Ludwig van Beethoven")
    ds.graph.graph.nodes["e1"]["birth_date"] = "2024-01-01"
    ds.graph.graph.nodes["e2"]["birth_date"] = "2023-01-01"
    ds.graph.graph.add_edge("e1", "e2", relation="parent_of")
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")

    class DummyDriver:
        def close(self):
            pass

        def session(self):
            class S:
                def __enter__(self_s):
                    return self_s

                def __exit__(self_s, exc_type, exc, tb):
                    pass

                def execute_write(self_s, func, *a, **k):
                    func(self_s, *a, **k)

                def run(self_s, *a, **k):
                    pass

            return S()

    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: DummyDriver())
    client.sadd("user:1:datasets", "demo")
    DATASETS["demo"] = ds

    class FakeResponse:
        def __init__(self, data):
            self._data = data

        def json(self):
            return self._data

    def fake_get(url, params=None, headers=None, timeout=10):
        if "lookup.dbpedia.org" in url:
            return FakeResponse(
                {
                    "results": [
                        {
                            "id": "http://dbpedia.org/resource/Beethoven",
                            "description": "desc",
                        }
                    ]
                }
            )
        return FakeResponse({"search": [{"id": "Q1", "description": "composer"}]})

    with app.test_client() as cl:
        _login(cl)
        ops = [
            "consolidate",
            "communities",
            "summaries",
            "trust",
            "similarity",
            "co_mentions",
            "doc_co_mentions",
            "section_co_mentions",
            "author_org_links",
            "entity_groups",
            "entity_group_summaries",
            "deduplicate",
            "clean_chunks",
            "normalize_dates",
            "prune",
            "graph_embeddings",
            "mark_conflicts",
            "validate",
            "resolve_entities",
            "predict_links?graph=true",
            "centrality",
        ]
        for op in ops:
            if op == "prune":
                res = cl.post(f"/api/datasets/demo/{op}", json={"sources": ["bad"]})
            elif op == "resolve_entities":
                res = cl.post(
                    f"/api/datasets/demo/{op}",
                    json={"aliases": {"Beethoven": ["Ludwig van Beethoven"]}},
                )
            elif op == "graph_embeddings":
                res = cl.post(
                    f"/api/datasets/demo/{op}",
                    json={
                        "dimensions": 8,
                        "walk_length": 4,
                        "num_walks": 5,
                        "seed": 42,
                        "workers": 1,
                    },
                )
            else:
                res = cl.post(f"/api/datasets/demo/{op}")
            assert res.status_code == 200
        ds = DatasetBuilder.from_redis(client, "dataset:demo")
        assert "d2" not in ds.graph.graph.nodes
        assert "c3" not in ds.graph.graph.nodes
        assert len(ds.graph.graph.nodes["e1"]["embedding"]) == 8

        monkeypatch.setattr(requests, "get", fake_get)
        res = cl.post("/api/datasets/demo/enrich_entity/e1")
        assert res.status_code == 200
        res = cl.post("/api/datasets/demo/enrich_entity_dbpedia/e1")
        assert res.status_code == 200
        res = cl.post("/api/datasets/demo/extract_entities", json={"model": None})
        assert res.status_code == 200
        ds = DatasetBuilder.from_redis(client, "dataset:demo")
        assert any(e.operation == "enrich_entity_dbpedia" for e in ds.events)
    DATASETS.clear()


def test_dataset_owner_isolation(monkeypatch):
    client = fakeredis.FakeStrictRedis()
    app_module.REDIS = client
    app_module.DATASETS.clear()

    with app.test_client() as cl:
        # login as alice (user id 1) and create dataset
        cl.post("/api/login", json={"username": "alice", "password": "pw"})
        res = cl.post("/api/datasets", json={"name": "demo", "dataset_type": "qa"})
        assert res.status_code == 200
        cl.get("/api/logout")

        # register second user and login
        cl.post("/api/register", json={"username": "bob2", "password": "pw"})
        cl.post("/api/login", json={"username": "bob2", "password": "pw"})
        res = cl.get("/api/datasets/demo")
        assert res.status_code == 404


def test_dataset_list_visibility(monkeypatch):
    client = fakeredis.FakeStrictRedis()
    app_module.REDIS = client
    app_module.DATASETS.clear()

    with app.test_client() as cl:
        cl.post("/api/login", json={"username": "alice", "password": "pw"})
        cl.post("/api/datasets", json={"name": "demo", "dataset_type": "qa"})
        cl.get("/api/logout")

        cl.post("/api/register", json={"username": "bob3", "password": "pw"})
        cl.post("/api/login", json={"username": "bob3", "password": "pw"})
        res = cl.get("/api/datasets")
        assert res.status_code == 200
        assert "demo" not in res.get_json()


def test_lookup_endpoints():
    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.add_document("doc", source="s")
    ds.add_section("doc", "sec1")
    ds.add_chunk("doc", "c1", "A is B", section_id="sec1")
    ds.add_entity("A", "A")
    ds.add_entity("B", "B")
    ds.link_entity("c1", "A")
    ds.link_entity("c1", "B")
    fid = ds.graph.add_fact("A", "is", "B")
    ds.graph.graph.add_edge("c1", fid, relation="has_fact")
    DATASETS["demo"] = ds

    with app.test_client() as client:
        _login(client)
        res = client.get(
            "/api/datasets/demo/chunk_document", query_string={"cid": "c1"}
        )
        assert res.status_code == 200
        assert res.get_json() == "doc"

        res = client.get("/api/datasets/demo/chunk_page", query_string={"cid": "c1"})
        assert res.status_code == 200
        assert res.get_json() == 1

        res = client.get(
            "/api/datasets/demo/section_page",
            query_string={"sid": "sec1"},
        )
        assert res.status_code == 200
        assert res.get_json() == 1

        res = client.get(
            "/api/datasets/demo/section_document", query_string={"sid": "sec1"}
        )
        assert res.status_code == 200
        assert res.get_json() == "doc"

        res = client.get(
            "/api/datasets/demo/chunk_entities", query_string={"cid": "c1"}
        )
        assert set(res.get_json()) == {"A", "B"}

        res = client.get("/api/datasets/demo/chunk_facts", query_string={"cid": "c1"})
        assert res.get_json() == [fid]

        res = client.get("/api/datasets/demo/fact_sections", query_string={"fid": fid})
        assert res.get_json() == ["sec1"]

        res = client.get("/api/datasets/demo/fact_documents", query_string={"fid": fid})
        assert res.get_json() == ["doc"]

        res = client.get("/api/datasets/demo/fact_pages", query_string={"fid": fid})
        assert res.get_json() == [1]

        res = client.get(
            "/api/datasets/demo/entity_documents", query_string={"eid": "A"}
        )
        assert res.get_json() == ["doc"]

        res = client.get("/api/datasets/demo/entity_chunks", query_string={"eid": "A"})
        assert res.get_json() == ["c1"]

        res = client.get("/api/datasets/demo/entity_facts", query_string={"eid": "A"})
        assert res.get_json() == [fid]

        res = client.get("/api/datasets/demo/entity_pages", query_string={"eid": "A"})
        assert res.get_json() == [1]
    DATASETS.clear()


def test_conflicts_endpoint():
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.graph.add_fact("A", "likes", "B", source="s1")
    ds.graph.add_fact("A", "likes", "C", source="s2")
    DATASETS["demo"] = ds
    with app.test_client() as client:
        _login(client)
        res = client.get("/api/datasets/demo/conflicts")
        assert res.status_code == 200
        data = res.get_json()
        assert data == [["A", "likes", {"B": ["s1"], "C": ["s2"]}]]
    DATASETS.clear()


def test_api_delete_dataset(monkeypatch):
    client = fakeredis.FakeStrictRedis()
    app_module.REDIS = client
    monkeypatch.setattr("datacreek.tasks.get_redis_client", lambda: client)
    import datacreek.tasks as tasks_mod

    tasks_mod.celery_app.conf.task_always_eager = True

    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.add_document("d", source="s")
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")

    class DummyDriver:
        def close(self):
            pass

        def session(self):
            class S:
                def __enter__(self_s):
                    return self_s

                def __exit__(self_s, exc_type, exc, tb):
                    pass

                def run(self_s, *a, **k):
                    pass

            return S()

    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: DummyDriver())

    with app.test_client() as cl:
        _login(cl)
        res = cl.delete("/api/datasets/demo")
        assert res.status_code == 200
        data = res.get_json()
        assert "task_id" in data

    prog = json.loads(client.hget("dataset:demo:progress", "delete"))
    assert prog.get("deleted") is True
    assert "time" in prog
    assert client.hget("dataset:demo:progress", "delete_start") is not None
    assert client.hget("dataset:demo:progress", "delete_finish") is not None

    assert not client.exists("dataset:demo")
    assert "demo" not in client.smembers("datasets")


def test_delete_dataset_unauthorized(monkeypatch):
    client = fakeredis.FakeStrictRedis()
    app_module.REDIS = client
    monkeypatch.setattr("datacreek.tasks.get_redis_client", lambda: client)
    import datacreek.tasks as tasks_mod

    tasks_mod.celery_app.conf.task_always_eager = True

    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.owner_id = 1
    ds.add_document("d", source="s")
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")
    client.sadd("user:1:datasets", "demo")

    with app.test_client() as cl:
        cl.post("/api/register", json={"username": "bob", "password": "pw"})
        cl.post("/api/login", json={"username": "bob", "password": "pw"})
        res = cl.delete("/api/datasets/demo")
        assert res.status_code == 404


def test_graph_api(monkeypatch, tmp_path):
    client = fakeredis.FakeStrictRedis()
    app_module.REDIS = client
    monkeypatch.setattr("datacreek.tasks.get_redis_client", lambda: client)

    class DummyDriver:
        def close(self):
            pass

        def session(self):
            class S:
                def __enter__(self_s):
                    return self_s

                def __exit__(self_s, exc_type, exc, tb):
                    pass

                def run(self_s, *a, **k):
                    pass

                def execute_write(self_s, func, *a, **k):
                    pass

            return S()

    monkeypatch.setattr(app_module, "get_neo4j_driver", lambda: DummyDriver())
    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: DummyDriver())
    monkeypatch.setattr(app_module.KnowledgeGraph, "to_neo4j", lambda *a, **k: None)
    monkeypatch.setattr(
        app_module.KnowledgeGraph,
        "from_neo4j",
        staticmethod(lambda *a, **k: app_module.KnowledgeGraph()),
    )
    import datacreek.tasks as tasks_mod

    tasks_mod.celery_app.conf.task_always_eager = True

    f = tmp_path / "doc.txt"
    f.write_text("hello world")

    with app.test_client() as cl:
        _login(cl)
        res = cl.post("/api/graphs", json={"name": "g", "documents": [str(f)]})
        assert res.status_code == 200
        res = cl.get("/api/graphs")
        assert res.get_json() == ["g"]
        res = cl.get("/api/graphs/g")
        assert res.status_code == 200
        res = cl.get("/api/graphs/g/data")
        assert res.status_code == 200
        res = cl.post("/api/graphs/g/save_neo4j")
        assert res.status_code == 200
        res = cl.get("/api/graphs/g/progress")
        data = res.get_json()
        assert "time" in data.get("save_neo4j")
        assert data.get("save_neo4j_start") is not None
        assert data.get("save_neo4j_finish") is not None
        res = cl.post("/api/graphs/g/load_neo4j")
        assert res.status_code == 200
        res = cl.get("/api/graphs/g/progress")
        data = res.get_json()
        assert "time" in data.get("load_neo4j")
        assert data.get("load_neo4j_start") is not None
        assert data.get("load_neo4j_finish") is not None
        res = cl.delete("/api/graphs/g")
        assert res.status_code == 200
        res = cl.get("/api/graphs/g/progress")
        assert res.status_code == 404

    assert "g" not in client.smembers("graphs")


def test_export_result_endpoint(monkeypatch):
    client = fakeredis.FakeStrictRedis()
    app_module.REDIS = client
    monkeypatch.setattr("datacreek.tasks.get_redis_client", lambda: client)
    import datacreek.tasks as tasks_mod

    tasks_mod.celery_app.conf.task_always_eager = True

    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.redis_client = client
    ds.add_document("d1", source="s")
    ds.add_chunk("d1", "c1", "hello")
    ds.versions.append({"result": {"qa_pairs": [{"question": "q", "answer": "a"}]}})
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")

    dataset_export_task.delay("demo", ExportFormat.JSONL).get()
    with app.test_client() as cl:
        _login(cl)
        res = cl.get("/api/datasets/demo/export_result", query_string={"fmt": "jsonl"})
        assert res.status_code == 200
        data = res.data.decode()
        assert "question" in data and "answer" in data
        prog = cl.get("/api/datasets/demo/progress").get_json()
        assert prog.get("export", {}).get("fmt") == "jsonl"
        assert "time" in prog.get("export", {})
    DATASETS.clear()


def test_dataset_version_endpoints(monkeypatch):
    client = fakeredis.FakeStrictRedis()
    app_module.REDIS = client
    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.redis_client = client
    ds.versions.append({"params": {"n": 1}, "time": "t", "result": {"qa_pairs": [1]}})
    ds.versions.append({"params": {"n": 2}, "time": "t2", "result": {"qa_pairs": [2]}})
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")

    with app.test_client() as cl:
        _login(cl)
        res = cl.get("/api/datasets/demo/versions")
        assert res.status_code == 200
        versions = res.get_json()
        assert len(versions) == 2

        res = cl.get("/api/datasets/demo/versions/2")
        assert res.status_code == 200
        ver = res.get_json()
        assert ver["params"]["n"] == 2
        assert ver["result"]["qa_pairs"] == [2]

    DATASETS.clear()


def test_dataset_progress_endpoint(monkeypatch):
    client = fakeredis.FakeStrictRedis()
    app_module.REDIS = client
    monkeypatch.setattr("datacreek.tasks.get_redis_client", lambda: client)
    import datacreek.tasks as tasks_mod

    tasks_mod.celery_app.conf.task_always_eager = True

    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.redis_client = client
    ds.add_document("d1", source="s", text="t")
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")

    def fake_run(self, redis_client=None, **p):
        self.stage = 2
        if redis_client is not None:
            redis_client.hset("dataset:demo:progress", "generate_qa_duration", "0.0")
        self._record_event("generate", "ok")

    monkeypatch.setattr(DatasetBuilder, "run_post_kg_pipeline", fake_run)

    tasks_mod.dataset_generate_task.delay("demo", {"start_step": "CURATE"}).get()

    with app.test_client() as cl:
        _login(cl)
        res = cl.get("/api/datasets/demo/progress")
        assert res.status_code == 200
        data = res.get_json()
        assert data.get("generation_params", {}).get("start_step") == "CURATE"
        assert data.get("generated_version") == len(
            DatasetBuilder.from_redis(client, "dataset:demo").versions
        )
        assert "generate_start" in data
        assert "generate_finish" in data
        assert "generate_qa_duration" in data

    DATASETS.clear()


def test_dataset_history_endpoint(monkeypatch):
    client = fakeredis.FakeStrictRedis()
    app_module.REDIS = client
    monkeypatch.setattr("datacreek.tasks.get_redis_client", lambda: client)

    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.add_document("d1", source="s")
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")

    with app.test_client() as cl:
        _login(cl)
        res = cl.get("/api/datasets/demo/history")
        assert res.status_code == 200
        data = res.get_json()
        assert any(ev["operation"] == "add_document" for ev in data)
        assert client.llen("dataset:demo:events") > 0

    DATASETS.clear()


def test_cleanup_progress(monkeypatch):
    client = fakeredis.FakeStrictRedis()
    app_module.REDIS = client
    monkeypatch.setattr("datacreek.tasks.get_redis_client", lambda: client)
    import datacreek.tasks as tasks_mod

    tasks_mod.celery_app.conf.task_always_eager = True

    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.redis_client = client
    ds.add_document("d1", source="s")
    ds.add_chunk("d1", "c1", "dup")
    ds.add_chunk("d1", "c2", "dup")
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")

    tasks_mod.dataset_cleanup_task.delay("demo", {}).get()

    with app.test_client() as cl:
        _login(cl)
        res = cl.get("/api/datasets/demo/progress")
        assert res.status_code == 200
        data = res.get_json()
        assert data.get("cleanup", {}).get("removed") == 1
        assert "time" in data.get("cleanup", {})
        assert "cleanup_start" in data
        assert "cleanup_finish" in data

    DATASETS.clear()


def test_graph_access_unauthorized(monkeypatch):
    client = fakeredis.FakeStrictRedis()
    app_module.REDIS = client
    monkeypatch.setattr("datacreek.tasks.get_redis_client", lambda: client)
    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: None)
    import datacreek.tasks as tasks_mod

    tasks_mod.celery_app.conf.task_always_eager = True

    ds = DatasetBuilder(DatasetType.TEXT, name="g")
    ds.redis_client = client
    ds.owner_id = 1
    ds.add_document("d", source="s")
    ds.to_redis(client, "graph:g")
    client.sadd("graphs", "g")
    client.sadd("user:1:graphs", "g")

    with app.test_client() as cl:
        cl.post("/api/register", json={"username": "bob", "password": "pw"})
        cl.post("/api/login", json={"username": "bob", "password": "pw"})
        res = cl.get("/api/graphs/g")
        assert res.status_code == 404
        res = cl.get("/api/graphs/g/data")
        assert res.status_code == 404
        res = cl.delete("/api/graphs/g")
        assert res.status_code == 404
        res = cl.post("/api/graphs/g/save_neo4j")
        assert res.status_code == 404
        res = cl.post("/api/graphs/g/load_neo4j")
        assert res.status_code == 404


def test_graph_list_visibility(monkeypatch, tmp_path):
    client = fakeredis.FakeStrictRedis()
    app_module.REDIS = client
    monkeypatch.setattr("datacreek.tasks.get_redis_client", lambda: client)
    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: None)
    import datacreek.tasks as tasks_mod

    tasks_mod.celery_app.conf.task_always_eager = True

    f = tmp_path / "doc.txt"
    f.write_text("hello")

    with app.test_client() as cl:
        # create graph as alice
        _login(cl)
        cl.post("/api/graphs", json={"name": "g1", "documents": [str(f)]})
        cl.get("/api/logout")

        # register bob and check listing
        cl.post("/api/register", json={"username": "bob4", "password": "pw"})
        cl.post("/api/login", json={"username": "bob4", "password": "pw"})
        res = cl.get("/api/graphs")
        assert res.status_code == 200
        assert "g1" not in res.get_json()


def test_dataset_from_foreign_graph_denied(monkeypatch, tmp_path):
    client = fakeredis.FakeStrictRedis()
    app_module.REDIS = client
    monkeypatch.setattr("datacreek.tasks.get_redis_client", lambda: client)
    monkeypatch.setattr(app_module, "get_neo4j_driver", lambda: None)
    import datacreek.tasks as tasks_mod

    tasks_mod.celery_app.conf.task_always_eager = True

    f = tmp_path / "doc.txt"
    f.write_text("hello")

    with app.test_client() as cl:
        # alice creates a graph
        _login(cl)
        cl.post("/api/graphs", json={"name": "g2", "documents": [str(f)]})
        cl.get("/api/logout")

        # bob tries to create dataset from alice's graph
        cl.post("/api/register", json={"username": "bob5", "password": "pw"})
        cl.post("/api/login", json={"username": "bob5", "password": "pw"})
        res = cl.post(
            "/api/datasets",
            json={"name": "ds", "dataset_type": "qa", "graph": "g2"},
        )
        assert res.status_code == 404


def test_multiple_graphs_per_user(monkeypatch):
    client = fakeredis.FakeStrictRedis()
    app_module.REDIS = client
    monkeypatch.setattr("datacreek.tasks.get_redis_client", lambda: client)
    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: None)
    import datacreek.tasks as tasks_mod

    tasks_mod.celery_app.conf.task_always_eager = True

    g1 = DatasetBuilder(DatasetType.TEXT, name="g1")
    g1.owner_id = 1
    g1.redis_client = client
    g1.to_redis(client, "graph:g1")
    client.sadd("graphs", "g1")
    client.sadd("user:1:graphs", "g1")

    g2 = DatasetBuilder(DatasetType.TEXT, name="g2")
    g2.owner_id = 1
    g2.redis_client = client
    g2.to_redis(client, "graph:g2")
    client.sadd("graphs", "g2")
    client.sadd("user:1:graphs", "g2")

    with app.test_client() as cl:
        _login(cl)
        res = cl.get("/api/graphs")
        assert res.status_code == 200
        assert set(res.get_json()) == {"g1", "g2"}


def test_api_explain_node():
    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.add_document("d1", source="s")
    ds.add_chunk("d1", "c1", "a")
    ds.add_chunk("d1", "c2", "b")
    ds.graph.graph.add_edge("c1", "c2", relation="similar_to")
    ds.graph.graph.nodes["c1"]["embedding"] = [0.0, 0.0]
    ds.graph.graph.nodes["c2"]["embedding"] = [1.0, 0.0]
    DATASETS["demo"] = ds

    with app.test_client() as client:
        _login(client)
        res = client.get("/api/datasets/demo/explain/c1")
        assert res.status_code == 200
        data = res.get_json()
        assert "nodes" in data and "edges" in data and "attention" in data
        assert "c1" in data["nodes"] and "c2" in data["nodes"]
    DATASETS.clear()
