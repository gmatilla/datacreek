import numpy as np
import pytest

from datacreek.analysis import index


def test_search_with_fallback(monkeypatch):
    if index.faiss is None:
        pytest.skip("faiss not installed")
    xb = np.eye(2, dtype=np.float32)
    xq = np.asarray([[1.0, 0.0]], dtype=np.float32)
    idx, latency, _ = index.search_with_fallback(xb, xq, k=1, latency_threshold=0.0)
    assert idx[0] == 0


def test_recall10_property():
    import networkx as nx

    G = nx.Graph()
    G.add_node(
        "a",
        embedding=[1.0, 0.0],
        graphwave_embedding=[1.0, 0.0],
        poincare_embedding=[1.0, 0.0],
    )
    G.add_node(
        "b",
        embedding=[0.0, 1.0],
        graphwave_embedding=[0.0, 1.0],
        poincare_embedding=[0.0, 1.0],
    )
    score = index.recall10(G, ["a"], {"a": ["b"]})
    assert 0.0 <= score <= 1.0
    assert G.graph.get("recall10") == score
    if index.recall_gauge is not None:
        assert index.recall_gauge._value.get() == score
