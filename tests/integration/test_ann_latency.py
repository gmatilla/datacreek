import sys
import types

from prometheus_client import Histogram

from datacreek.core.knowledge_graph import KnowledgeGraph

# Provide dummy sklearn to avoid heavy dependency
sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
cross = types.ModuleType("sklearn.cross_decomposition")
cross.CCA = object
sys.modules.setdefault("sklearn.cross_decomposition", cross)
import datacreek.analysis.index as idx


def test_ann_latency_histogram(monkeypatch):
    kg = KnowledgeGraph()
    kg.graph.add_node("a", embedding=[1.0, 0.0])
    kg.graph.add_node("b", embedding=[0.0, 1.0])
    hist = Histogram("ann_latency_seconds_test", "lat", buckets=(5, 10))
    monkeypatch.setattr(idx, "ann_latency", hist, raising=False)
    monkeypatch.setattr(
        "datacreek.core.knowledge_graph.ann_latency", hist, raising=False
    )
    kg.build_faiss_index()
    kg.search_faiss([1.0, 0.0], k=1)
    buckets = {
        float(s.labels["le"]): s.value
        for s in hist.collect()[0].samples
        if "_bucket" in s.name
    }
    assert buckets.get(5.0, 0) > 0 or buckets.get(10.0, 0) > 0
