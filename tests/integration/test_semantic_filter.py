import networkx as nx

from datacreek.analysis.filtering import filter_semantic_cycles
from datacreek.core.dataset import DatasetBuilder
from datacreek.pipelines import DatasetType


def test_filter_semantic_cycles():
    g = nx.DiGraph()
    g.add_node("a", text="the")
    g.add_node("b", text="and")
    g.add_node("c", text="of")
    g.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
    filtered = filter_semantic_cycles(g, max_len=3)
    assert filtered.number_of_edges() == 0


def test_dataset_filter_semantic_cycles():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.graph.graph.add_node("a", text="the")
    ds.graph.graph.add_node("b", text="and")
    ds.graph.graph.add_node("c", text="of")
    ds.graph.graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
    ds.filter_semantic_cycles(max_len=3)
    assert ds.graph.graph.number_of_edges() == 0
    assert any(e.operation == "filter_semantic_cycles" for e in ds.events)
