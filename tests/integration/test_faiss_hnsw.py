import os
import sys
import types

import pytest

faiss = pytest.importorskip("faiss")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datacreek.core.knowledge_graph import KnowledgeGraph


@pytest.mark.faiss_gpu
def test_faiss_adaptive_switch(monkeypatch):
    kg = KnowledgeGraph()
    kg.graph.add_node("a", embedding=[1.0, 0.0])
    kg.graph.add_node("b", embedding=[0.0, 1.0])
    try:
        kg.build_faiss_index()
    except RuntimeError:
        pytest.skip("faiss not installed")

    times = [0.0, 0.0]

    def fake_monotonic():
        t = times.pop(0)
        return t

    monkeypatch.setattr("time.monotonic", fake_monotonic)
    # monkeypatch the second call to time to simulate 0.2s later
    times.append(0.2)

    res = kg.search_faiss([1.0, 0.0], k=1, adaptive=True, latency_threshold=0.1)
    assert res == ["a"]
    assert kg.faiss_index_type == "hnsw"
