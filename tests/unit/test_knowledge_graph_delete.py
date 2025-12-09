import types

import numpy as np

from datacreek.core.knowledge_graph import KnowledgeGraph


class DummySession:
    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def run(self, q, **params):
        self.calls["query"] = q
        self.calls.update(params)


class DummyDriver:
    def __init__(self, calls):
        self.calls = calls

    def session(self):
        return DummySession(self.calls)


class DummyFaiss:
    def __init__(self):
        self.removed = []
        self.added = []

    def remove_ids(self, idxs):
        self.removed.extend(idxs.tolist())

    def add(self, vec):
        self.added.append(vec.tolist())


def test_cascade_delete_document(monkeypatch):
    kg = KnowledgeGraph()
    kg.add_document("d", "src")
    kg.graph.nodes["d"]["embedding"] = np.array([1.0, 0.0], dtype="float32")
    kg.faiss_index = DummyFaiss()
    kg.faiss_ids = ["d"]
    kg.faiss_node_attr = "embedding"
    calls = {}
    driver = DummyDriver(calls)

    kg.cascade_delete_document("d", driver=driver)

    assert calls["uid"] == "d"
    assert kg.faiss_ids == ["doc:deleted:d"]
    assert kg.faiss_index.removed == [0]
    assert kg.faiss_index.added
