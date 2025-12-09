import networkx as nx

from datacreek.analysis.filtering import entropy_triangle_threshold

try:
    from datacreek.core.dataset import DatasetBuilder
    from datacreek.core.knowledge_graph import KnowledgeGraph
except Exception:  # pragma: no cover - skip if heavy deps missing
    KnowledgeGraph = None  # type: ignore
    DatasetBuilder = None  # type: ignore


def test_entropy_triangle_threshold_basic():
    g = nx.Graph()
    g.add_edge("a", "b", weight=0.5)
    g.add_edge("b", "c", weight=0.5)
    g.add_edge("c", "a", weight=0.5)
    tau = entropy_triangle_threshold(g)
    assert tau >= 1


def test_adaptive_triangle_threshold_wrapper():
    if KnowledgeGraph is None:
        import pytest

        pytest.skip("KnowledgeGraph dependencies missing")
    kg = KnowledgeGraph()
    kg.graph.add_edge("a", "b", weight=1.0)
    kg.graph.add_edge("b", "c", weight=1.0)
    ds = DatasetBuilder(kg)
    val = ds.adaptive_triangle_threshold()
    assert isinstance(val, int) and val >= 1
