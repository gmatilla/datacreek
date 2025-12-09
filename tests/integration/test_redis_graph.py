import json
import os

import fakeredis
import pytest

from datacreek.core.dataset import DatasetBuilder, DatasetType
from datacreek.models.task_status import TaskStatus
from datacreek.tasks import (
    dataset_load_redis_graph_task,
    dataset_save_redis_graph_task,
    get_redis_client,
)


def setup_fake(monkeypatch):
    import datacreek.tasks as tasks_mod

    client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.tasks.get_redis_client", lambda: client)
    tasks_mod.celery_app.conf.task_always_eager = True
    return client


class DummyNode:
    def __init__(self, label=None, properties=None):
        self.label = label
        self.properties = properties or {}
        self.alias = None


class DummyEdge:
    def __init__(self, src, dst, relation, properties=None):
        self.src = src
        self.dst = dst
        self.relation = relation
        self.properties = properties or {}


class DummyGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.deleted = False
        self.committed = False
        self.queries = []

    def delete(self):
        self.deleted = True

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, edge):
        self.edges.append(edge)

    def commit(self):
        self.committed = True

    def query(self, q, params=None):
        self.queries.append(q)
        if "RETURN n.id" in q:
            return type(
                "R", (), {"result_set": [["d", "Document", {"dataset": "demo"}]]}
            )
        return type("R", (), {"result_set": [["d", "c1", "REL", {"dataset": "demo"}]]})


def test_dataset_redis_graph_tasks(monkeypatch):
    client = setup_fake(monkeypatch)

    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello")
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")

    dummy = DummyGraph()
    monkeypatch.setattr("datacreek.backends.get_redis_graph", lambda name: dummy)
    monkeypatch.setattr("datacreek.core.dataset.get_redis_graph", lambda name: dummy)
    monkeypatch.setattr("datacreek.tasks.get_redis_graph", lambda name: dummy)
    monkeypatch.setattr("datacreek.tasks.get_redis_graph", lambda name: dummy)
    monkeypatch.setattr("datacreek.core.dataset.RGNode", DummyNode)
    monkeypatch.setattr("datacreek.core.dataset.RGEdge", DummyEdge)
    monkeypatch.setenv("USE_REDIS_GRAPH", "1")

    dataset_save_redis_graph_task.delay("demo").get()
    assert dummy.queries
    prog = json.loads(client.hget("dataset:demo:progress", "save_redis_graph"))
    assert prog["nodes"] == len(ds.graph.graph)
    assert (
        client.hget("dataset:demo:progress", "status").decode()
        == TaskStatus.COMPLETED.value
    )


def test_persist_uses_redis_graph(monkeypatch):
    client = setup_fake(monkeypatch)

    dummy = DummyGraph()
    monkeypatch.setattr("datacreek.backends.get_redis_graph", lambda name: dummy)
    monkeypatch.setattr("datacreek.core.dataset.get_redis_graph", lambda name: dummy)
    monkeypatch.setenv("USE_REDIS_GRAPH", "1")
    monkeypatch.setattr("datacreek.core.dataset.RGNode", DummyNode)
    monkeypatch.setattr("datacreek.core.dataset.RGEdge", DummyEdge)

    monkeypatch.setenv("DATACREEK_REQUIRE_PERSISTENCE", "0")
    ds = DatasetBuilder(
        DatasetType.TEXT, name="demo", redis_client=client, neo4j_driver=None
    )
    ds.add_document("d", source="s")
    ds._persist()
    assert dummy.queries
