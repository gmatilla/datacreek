import networkx as nx
import pytest

import datacreek.analysis.filtering as filt


@pytest.mark.heavy
def test_filter_semantic_cycles_removal():
    g = nx.cycle_graph(3)
    for n, label in zip(g.nodes(), ["the", "and", "a"]):
        g.nodes[n]["text"] = label
    cleaned = filt.filter_semantic_cycles(g, stopwords={"the", "and", "a"}, max_len=3)
    assert cleaned.number_of_edges() == 0


@pytest.mark.heavy
def test_entropy_triangle_threshold():
    g = nx.Graph()
    g.add_edge(0, 1, weight=1.0)
    g.add_edge(1, 2, weight=2.0)
    g.add_edge(2, 0, weight=3.0)
    thresh = filt.entropy_triangle_threshold(g, weight="weight", base=2.0, scale=5.0)
    assert isinstance(thresh, int) and thresh >= 1
