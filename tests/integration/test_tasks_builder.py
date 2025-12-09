import json
import os

import fakeredis
import pytest

from datacreek.core.dataset import DatasetBuilder, DatasetType
from datacreek.models.export_format import ExportFormat
from datacreek.models.stage import DatasetStage
from datacreek.models.task_status import TaskStatus
from datacreek.tasks import (
    dataset_cleanup_task,
    dataset_delete_task,
    dataset_delete_version_task,
    dataset_export_task,
    dataset_extract_entities_task,
    dataset_extract_facts_task,
    dataset_generate_task,
    dataset_ingest_task,
    dataset_load_neo4j_task,
    dataset_operation_task,
    dataset_prune_versions_task,
    dataset_restore_version_task,
    dataset_save_neo4j_task,
    datasets_prune_versions_task,
    get_redis_client,
    graph_delete_task,
    graph_load_neo4j_task,
    graph_save_neo4j_task,
)


def setup_fake(monkeypatch):
    import datacreek.tasks as tasks_mod

    client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.tasks.get_redis_client", lambda: client)
    tasks_mod.celery_app.conf.task_always_eager = True
    return client


def test_dataset_ingest_task(tmp_path, monkeypatch):
    client = setup_fake(monkeypatch)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")
    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: None)
    f = tmp_path / "doc.txt"
    f.write_text("hello world")
    dataset_ingest_task.delay(
        "demo",
        str(f),
        high_res=True,
        ocr=True,
        extract_entities=True,
    ).get()
    loaded = DatasetBuilder.from_redis(client, "dataset:demo")
    assert loaded.search("hello") == ["doc_chunk_0"]
    assert loaded.events[-1].operation == "ingest_document"
    assert int(client.hget("dataset:demo:progress", "ingested")) == 1
    last = json.loads(client.hget("dataset:demo:progress", "last_ingested"))
    assert last["path"] == str(f)
    assert "time" in last
    assert int(client.hget("dataset:demo:progress", "ingested_chunks")) >= 1
    assert client.hget("dataset:demo:progress", "ingest_start") is not None
    assert client.hget("dataset:demo:progress", "ingest_finish") is not None
    params = json.loads(client.hget("dataset:demo:progress", "ingestion_params"))
    assert params == {"high_res": True, "ocr": True, "extract_entities": True}
    assert (
        client.hget("dataset:demo:progress", "status").decode()
        == TaskStatus.COMPLETED.value
    )
    hist = [
        json.loads(x) for x in client.lrange("dataset:demo:progress:history", 0, -1)
    ]
    assert hist[0]["status"] == TaskStatus.INGESTING.value
    assert hist[-1]["status"] == TaskStatus.COMPLETED.value


def test_dataset_ingest_task_async(tmp_path, monkeypatch):
    client = setup_fake(monkeypatch)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")
    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: None)
    f = tmp_path / "doc.txt"
    f.write_text("hello async")
    dataset_ingest_task.delay("demo", str(f), async_mode=True).get()
    loaded = DatasetBuilder.from_redis(client, "dataset:demo")
    assert loaded.search("hello") == ["doc_chunk_0"]


def test_dataset_generate_task(monkeypatch):
    client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.tasks.get_redis_client", lambda: client)
    os.environ["CELERY_TASK_ALWAYS_EAGER"] = "1"
    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.redis_client = client
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "text")
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")

    called = {}

    def fake_run(self, redis_client=None, **params):
        called["ok"] = True
        if redis_client is not None:
            redis_client.hset("dataset:demo:progress", "generate_qa_duration", "0.0")
        self.stage = DatasetStage.GENERATED
        self._record_event("generate", "done")
        return {}

    monkeypatch.setattr(DatasetBuilder, "run_post_kg_pipeline", fake_run)
    dataset_generate_task.delay("demo", {"start_step": "CURATE"}).get()
    loaded = DatasetBuilder.from_redis(client, "dataset:demo")
    assert called["ok"]
    assert loaded.stage == DatasetStage.GENERATED
    assert any(e.operation == "generate" for e in loaded.events)
    progress = client.hget("dataset:demo:progress", "generation_params")
    assert progress is not None
    assert json.loads(progress)["start_step"] == "CURATE"
    assert int(
        json.loads(client.hget("dataset:demo:progress", "generated_version"))
    ) == len(loaded.versions)
    assert client.hget("dataset:demo:progress", "generate_start") is not None
    assert client.hget("dataset:demo:progress", "generate_finish") is not None
    assert client.hget("dataset:demo:progress", "generate_qa_duration") is not None
    assert (
        client.hget("dataset:demo:progress", "status").decode()
        == TaskStatus.COMPLETED.value
    )


def test_dataset_cleanup_and_export_tasks(monkeypatch):
    client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.tasks.get_redis_client", lambda: client)
    os.environ["CELERY_TASK_ALWAYS_EAGER"] = "1"

    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "<b>Hello</b>")
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")

    dataset_cleanup_task.delay("demo", {"normalize_dates": False}).get()
    loaded = DatasetBuilder.from_redis(client, "dataset:demo")
    assert any(e.operation == "clean_chunks" for e in loaded.events)
    assert any(e.operation == "cleanup_graph" for e in loaded.events)
    clean_prog = json.loads(client.hget("dataset:demo:progress", "cleanup"))
    assert "time" in clean_prog and clean_prog["removed"] >= 0
    assert client.hget("dataset:demo:progress", "cleanup_start") is not None
    assert client.hget("dataset:demo:progress", "cleanup_finish") is not None
    assert (
        client.hget("dataset:demo:progress", "status").decode()
        == TaskStatus.COMPLETED.value
    )

    result = dataset_export_task.delay("demo", ExportFormat.JSONL).get()
    assert result["stage"] == DatasetStage.EXPORTED
    assert client.get(result["key"]) is not None
    loaded2 = DatasetBuilder.from_redis(client, "dataset:demo")
    assert loaded2.stage == DatasetStage.EXPORTED
    assert any(e.operation == "export_dataset" for e in loaded2.events)
    progress = json.loads(client.hget("dataset:demo:progress", "export"))
    assert progress["fmt"] == "jsonl"
    assert progress["key"] == result["key"]
    assert "time" in progress
    assert (
        client.hget("dataset:demo:progress", "status").decode()
        == TaskStatus.COMPLETED.value
    )


def test_dataset_save_and_load_neo4j_tasks(monkeypatch):
    client = setup_fake(monkeypatch)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.add_document("d", source="s")
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")

    called = {}

    class DummyDriver:
        def close(self):
            called.setdefault("close", 0)
            called["close"] += 1

    def fake_to_neo4j(driver, clear=True, dataset=None):
        called["save"] = True

    def fake_from_neo4j(driver, dataset=None):
        called["load"] = True
        kg = ds.graph.__class__()
        kg.add_document("n", source="a")
        return kg

    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: DummyDriver())
    monkeypatch.setattr(ds.graph.__class__, "to_neo4j", fake_to_neo4j)
    monkeypatch.setattr(ds.graph.__class__, "from_neo4j", staticmethod(fake_from_neo4j))

    dataset_save_neo4j_task.delay("demo").get()
    loaded = DatasetBuilder.from_redis(client, "dataset:demo")
    assert called.get("save")
    assert called.get("close") == 1
    assert any(e.operation == "save_neo4j" for e in loaded.events)
    save_prog = json.loads(client.hget("dataset:demo:progress", "save_neo4j"))
    assert save_prog["nodes"] == len(loaded.graph.graph)
    assert "time" in save_prog
    assert client.hget("dataset:demo:progress", "save_neo4j_start") is not None
    assert client.hget("dataset:demo:progress", "save_neo4j_finish") is not None
    assert (
        client.hget("dataset:demo:progress", "status").decode()
        == TaskStatus.COMPLETED.value
    )

    dataset_load_neo4j_task.delay("demo").get()
    loaded2 = DatasetBuilder.from_redis(client, "dataset:demo")
    assert called.get("load")
    assert loaded2.graph.search_documents("n") == ["n"]
    assert any(e.operation == "load_neo4j" for e in loaded2.events)
    load_prog = json.loads(client.hget("dataset:demo:progress", "load_neo4j"))
    assert load_prog["nodes"] == len(loaded2.graph.graph)
    assert "time" in load_prog
    assert client.hget("dataset:demo:progress", "load_neo4j_start") is not None
    assert client.hget("dataset:demo:progress", "load_neo4j_finish") is not None
    assert (
        client.hget("dataset:demo:progress", "status").decode()
        == TaskStatus.COMPLETED.value
    )


def test_dataset_delete_task(monkeypatch):
    client = setup_fake(monkeypatch)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.add_document("d", source="s")
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")

    called = {}

    class DummySession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def run(self, query, **params):
            called["run"] = params.get("dataset")

    class DummyDriver:
        def session(self):
            return DummySession()

        def close(self):
            called["close"] = called.get("close", 0) + 1

    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: DummyDriver())

    dataset_delete_task.delay("demo").get()

    assert not client.exists("dataset:demo")
    assert "demo" not in client.smembers("datasets")
    assert called.get("run") == "demo"
    assert called.get("close") == 1
    prog = json.loads(client.hget("dataset:demo:progress", "delete"))
    assert prog.get("deleted") is True
    assert "time" in prog
    assert client.hget("dataset:demo:progress", "delete_start") is not None
    assert client.hget("dataset:demo:progress", "delete_finish") is not None
    assert (
        client.hget("dataset:demo:progress", "status").decode()
        == TaskStatus.COMPLETED.value
    )


def test_dataset_delete_task_removes_redis_graph(monkeypatch):
    client = setup_fake(monkeypatch)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.add_document("d", source="s")
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")

    class DummyGraph:
        def __init__(self):
            self.queries = []

        def query(self, q, params=None):
            self.queries.append(q)

    dummy = DummyGraph()
    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: None)
    monkeypatch.setattr("datacreek.tasks.get_redis_graph", lambda name: dummy)
    monkeypatch.setattr("datacreek.core.dataset.get_redis_graph", lambda name: dummy)
    monkeypatch.setenv("USE_REDIS_GRAPH", "1")

    dataset_delete_task.delay("demo").get()

    assert not client.exists("dataset:demo")
    assert dummy.queries


def test_dataset_operation_task_progress(monkeypatch):
    client = setup_fake(monkeypatch)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "<b>Hello world</b>")
    ds.add_chunk("d", "c2", "Hello again")
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")

    dataset_operation_task.delay("demo", "clean_chunks").get()
    dataset_operation_task.delay("demo", "link_similar_chunks", {"k": 1}).get()

    loaded = DatasetBuilder.from_redis(client, "dataset:demo")
    assert any(e.operation == "clean_chunks" for e in loaded.events)
    prog = json.loads(client.hget("dataset:demo:progress", "clean_chunks"))
    assert prog["result"] >= 1
    assert "time" in prog
    assert (
        client.hget("dataset:demo:progress", "status").decode()
        == TaskStatus.COMPLETED.value
    )
    assert any(e.operation == "link_similar_chunks" for e in loaded.events)
    lprog = json.loads(client.hget("dataset:demo:progress", "link_similar_chunks"))
    assert "time" in lprog
    assert (
        client.hget("dataset:demo:progress", "operation").decode()
        == "link_similar_chunks"
    )


def test_graph_tasks(monkeypatch):
    client = setup_fake(monkeypatch)
    ds = DatasetBuilder(DatasetType.TEXT, name="graph")
    ds.redis_client = client
    ds.add_document("d", source="s")
    ds.to_redis(client, "graph:graph")
    client.sadd("graphs", "graph")

    called = {}

    class DummyDriver:
        def close(self):
            called["close"] = called.get("close", 0) + 1

        def session(self):
            class S:
                def __enter__(self_s):
                    return self_s

                def __exit__(self_s, exc_type, exc, tb):
                    pass

                def run(self_s, *a, **k):
                    called["run"] = True

            return S()

    def fake_to_neo4j(driver, clear=True, dataset=None):
        called["save"] = True

    def fake_from_neo4j(driver, dataset=None):
        called["load"] = True
        kg = ds.graph.__class__()
        kg.add_document("n", source="a")
        return kg

    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: DummyDriver())
    monkeypatch.setattr(ds.graph.__class__, "to_neo4j", fake_to_neo4j)
    monkeypatch.setattr(ds.graph.__class__, "from_neo4j", staticmethod(fake_from_neo4j))

    graph_save_neo4j_task.delay("graph", None).get()
    loaded = DatasetBuilder.from_redis(client, "graph:graph")
    assert called.get("save")
    assert called.get("close") == 1
    assert any(e.operation == "save_neo4j" for e in loaded.events)
    save_prog = json.loads(client.hget("graph:graph:progress", "save_neo4j"))
    assert save_prog["nodes"] == len(loaded.graph.graph)
    assert "time" in save_prog
    assert client.hget("graph:graph:progress", "save_neo4j_start") is not None
    assert client.hget("graph:graph:progress", "save_neo4j_finish") is not None
    assert (
        client.hget("graph:graph:progress", "status").decode()
        == TaskStatus.COMPLETED.value
    )

    graph_load_neo4j_task.delay("graph", None).get()
    loaded2 = DatasetBuilder.from_redis(client, "graph:graph")
    assert called.get("load")
    assert loaded2.graph.search_documents("n") == ["n"]
    assert any(e.operation == "load_neo4j" for e in loaded2.events)
    load_prog = json.loads(client.hget("graph:graph:progress", "load_neo4j"))
    assert load_prog["nodes"] == len(loaded2.graph.graph)
    assert "time" in load_prog
    assert client.hget("graph:graph:progress", "load_neo4j_start") is not None
    assert client.hget("graph:graph:progress", "load_neo4j_finish") is not None
    assert (
        client.hget("graph:graph:progress", "status").decode()
        == TaskStatus.COMPLETED.value
    )

    graph_delete_task.delay("graph", None).get()
    assert not client.exists("graph:graph")
    assert "graph" not in client.smembers("graphs")
    prog = json.loads(client.hget("graph:graph:progress", "delete"))
    assert prog.get("deleted") is True
    assert "time" in prog
    assert client.hget("graph:graph:progress", "delete_start") is not None
    assert client.hget("graph:graph:progress", "delete_finish") is not None
    assert (
        client.hget("graph:graph:progress", "status").decode()
        == TaskStatus.COMPLETED.value
    )


def test_graph_delete_filters_dataset(monkeypatch):
    client = setup_fake(monkeypatch)
    ds = DatasetBuilder(DatasetType.TEXT, name="graph")
    ds.redis_client = client
    ds.to_redis(client, "graph:graph")
    client.sadd("graphs", "graph")

    queries = {}

    class DummyDriver:
        def close(self):
            queries["closed"] = True

        def session(self):
            class S:
                def __enter__(self_s):
                    return self_s

                def __exit__(self_s, exc_type, exc, tb):
                    pass

                def run(self_s, query, **params):
                    queries["query"] = query
                    queries["params"] = params

            return S()

    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: DummyDriver())

    graph_delete_task.delay("graph", None).get()

    assert queries.get("query") == "MATCH (n {dataset:$dataset}) DETACH DELETE n"
    assert queries.get("params", {}).get("dataset") == "graph"


def test_extract_tasks_progress(monkeypatch):
    client = setup_fake(monkeypatch)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "Alice is president.")
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")

    dataset_extract_facts_task.delay("demo").get()
    dataset_extract_entities_task.delay("demo").get()

    f_prog = json.loads(client.hget("dataset:demo:progress", "extract_facts"))
    e_prog = json.loads(client.hget("dataset:demo:progress", "extract_entities"))
    assert f_prog["done"] is True
    assert e_prog["done"] is True
    assert "time" in f_prog and "time" in e_prog
    assert (
        client.hget("dataset:demo:progress", "status").decode()
        == TaskStatus.COMPLETED.value
    )


def test_graph_tasks_unauthorized(monkeypatch):
    client = setup_fake(monkeypatch)
    ds = DatasetBuilder(DatasetType.TEXT, name="graph")
    ds.owner_id = 1
    ds.redis_client = client
    ds.to_redis(client, "graph:graph")
    client.sadd("graphs", "graph")

    with pytest.raises(RuntimeError):
        graph_save_neo4j_task.delay("graph", 2).get()
    with pytest.raises(RuntimeError):
        graph_load_neo4j_task.delay("graph", 2).get()
    with pytest.raises(RuntimeError):
        graph_delete_task.delay("graph", 2).get()


def test_dataset_tasks_unauthorized(monkeypatch):
    client = setup_fake(monkeypatch)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.owner_id = 1
    ds.redis_client = client
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")

    with pytest.raises(RuntimeError):
        dataset_ingest_task.delay("demo", "/tmp/x", 2).get()
    with pytest.raises(RuntimeError):
        dataset_generate_task.delay("demo", None, 2).get()
    with pytest.raises(RuntimeError):
        dataset_delete_task.delay("demo", 2).get()


def test_task_error_history(monkeypatch):
    client = setup_fake(monkeypatch)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")

    def boom(*a, **k):
        raise RuntimeError("fail")

    monkeypatch.setattr(DatasetBuilder, "ingest_file", boom)

    with pytest.raises(RuntimeError):
        dataset_ingest_task.delay("demo", "bad", None).get()

    hist = [
        json.loads(x) for x in client.lrange("dataset:demo:progress:history", 0, -1)
    ]
    assert hist and hist[-1]["status"] == TaskStatus.FAILED.value
    assert "error" in hist[-1]


def test_task_error_event(monkeypatch):
    client = setup_fake(monkeypatch)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")

    def boom(*a, **k):
        raise RuntimeError("fail")

    monkeypatch.setattr(DatasetBuilder, "ingest_file", boom)

    with pytest.raises(RuntimeError):
        dataset_ingest_task.delay("demo", "bad", None).get()

    loaded = DatasetBuilder.from_redis(client, "dataset:demo")
    assert any(e.operation == "task_error" for e in loaded.events)


def test_dataset_prune_versions_task(monkeypatch):
    client = setup_fake(monkeypatch)
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.versions = [{"time": f"t{i}"} for i in range(5)]
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")

    dataset_prune_versions_task.delay("demo", 2).get()

    loaded = DatasetBuilder.from_redis(client, "dataset:demo")
    assert len(loaded.versions) == 2
    assert any(e.operation == "prune_versions" for e in loaded.events)


def test_datasets_prune_versions_task(monkeypatch):
    client = setup_fake(monkeypatch)
    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: None)
    for name in ["ds1", "ds2"]:
        ds = DatasetBuilder(DatasetType.TEXT, name=name)
        ds.redis_client = client
        ds.versions = [{"time": f"t{i}"} for i in range(4)]
        ds.to_redis(client, f"dataset:{name}")
        client.sadd("datasets", name)

    datasets_prune_versions_task.delay(2).get()

    for name in ["ds1", "ds2"]:
        loaded = DatasetBuilder.from_redis(client, f"dataset:{name}")
        assert len(loaded.versions) == 2


def test_dataset_restore_version_task(monkeypatch):
    client = setup_fake(monkeypatch)
    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.redis_client = client
    ds.versions = [
        {"time": "t1", "result": {"qa_pairs": [1]}},
        {"time": "t2", "result": {"qa_pairs": [2]}},
    ]
    ds.to_redis(client, "dataset:demo")

    dataset_restore_version_task.delay("demo", 1).get()

    loaded = DatasetBuilder.from_redis(client, "dataset:demo")
    assert len(loaded.versions) == 3
    assert loaded.versions[-1]["result"] == {"qa_pairs": [1]}
    assert any(e.operation == "restore_version" for e in loaded.events)


def test_dataset_delete_version_task(monkeypatch):
    client = setup_fake(monkeypatch)
    ds = DatasetBuilder(DatasetType.QA, name="demo")
    ds.redis_client = client
    ds.versions = [
        {"time": "t1", "result": {"qa_pairs": [1]}},
        {"time": "t2", "result": {"qa_pairs": [2]}},
    ]
    ds.to_redis(client, "dataset:demo")

    dataset_delete_version_task.delay("demo", 1).get()

    loaded = DatasetBuilder.from_redis(client, "dataset:demo")
    assert len(loaded.versions) == 1
    assert loaded.versions[0]["result"] == {"qa_pairs": [2]}
    assert any(e.operation == "delete_version" for e in loaded.events)
