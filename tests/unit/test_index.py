import builtins
import types

import networkx as nx
import numpy as np
import pytest

import datacreek.analysis.index as index


class DummyGauge:
    def __init__(self):
        self.val = None

    def set(self, val):
        self.val = val


class DummyFAISSIndex(
    index.faiss.IndexFlatIP if hasattr(index, "faiss") and index.faiss else object
):
    def __init__(self, dim):
        self.dim = dim
        self.xb = None
        self.hnsw = types.SimpleNamespace(efSearch=0)

    def add(self, xb):
        self.xb = xb

    def search(self, xq, k):
        sims = xq @ self.xb.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        return sims, idx


class DummyHNSW(DummyFAISSIndex):
    def __init__(self, dim, m):
        super().__init__(dim)

    pass


def test_search_with_fallback(monkeypatch):
    xb = np.eye(2, dtype="float32")
    xq = np.array([[1.0, 0.0]], dtype="float32")
    times = iter([0.0, 0.2, 0.2, 0.25])
    monkeypatch.setattr(
        index,
        "faiss",
        types.SimpleNamespace(IndexFlatIP=DummyFAISSIndex, IndexHNSWFlat=DummyHNSW),
    )
    monkeypatch.setattr(index, "np", np)
    monkeypatch.setattr(
        index, "time", types.SimpleNamespace(monotonic=lambda: next(times))
    )
    idx, lat, used = index.search_with_fallback(xb, xq, k=1, latency_threshold=0.1)
    assert idx == [0]
    assert isinstance(used, DummyHNSW)
    assert lat == pytest.approx(0.05)


def test_recall10(monkeypatch):
    g = nx.Graph()
    g.add_node(
        "a",
        embedding=np.array([1, 0]),
        graphwave_embedding=np.array([1, 0]),
        poincare_embedding=np.array([1, 0]),
    )
    g.add_node(
        "b",
        embedding=np.array([1, 0]),
        graphwave_embedding=np.array([1, 0]),
        poincare_embedding=np.array([1, 0]),
    )
    metrics = {}
    monkeypatch.setattr(index, "update_metric", lambda n, v: metrics.setdefault(n, v))
    monkeypatch.setattr(index, "recall_gauge", DummyGauge())
    monkeypatch.setattr(index, "hybrid_score", lambda *args, **kw: 1.0)
    r = index.recall10(g, ["a"], {"a": ["b"]})
    assert 0 < r <= 1
    assert metrics["recall10"] == r
    assert g.graph["recall10"] == r


def test_search_without_fallback(monkeypatch):
    xb = np.eye(2, dtype="float32")
    xq = np.array([[1.0, 0.0]], dtype="float32")
    times = iter([0.0, 0.05, 0.05, 0.1])
    monkeypatch.setattr(
        index,
        "faiss",
        types.SimpleNamespace(IndexFlatIP=DummyFAISSIndex, IndexHNSWFlat=DummyHNSW),
    )
    monkeypatch.setattr(index, "np", np)
    monkeypatch.setattr(
        index, "time", types.SimpleNamespace(monotonic=lambda: next(times))
    )
    idx, lat, used = index.search_with_fallback(xb, xq, k=1, latency_threshold=0.1)
    assert idx == [0]
    assert isinstance(used, DummyFAISSIndex)
    assert lat == pytest.approx(0.05)


def test_search_no_faiss(monkeypatch):
    monkeypatch.setattr(index, "faiss", None)
    monkeypatch.setattr(index, "np", np)
    with pytest.raises(RuntimeError):
        index.search_with_fallback(
            np.eye(1, dtype="float32"), np.ones((1, 1), dtype="float32")
        )


def test_recall10_no_hits(monkeypatch):
    g = nx.Graph()
    g.add_node("a", embedding=None)
    r = index.recall10(g, ["a"], {"a": ["b"]})
    assert r == 0
