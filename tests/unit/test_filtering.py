import networkx as nx
import numpy as np

from datacreek.analysis.filtering import (
    entropy_triangle_threshold,
    filter_semantic_cycles,
)


def test_filter_semantic_cycles_remove_stopword_cycle():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    for n, label in zip([1, 2, 3], ["the", "and", "of"]):
        G.nodes[n]["text"] = label
    filtered = filter_semantic_cycles(G)
    assert len(filtered.edges()) == 0


def test_filter_semantic_cycles_preserve_meaningful_cycle():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    for n, label in zip([1, 2, 3], ["alpha", "beta", "gamma"]):
        G.nodes[n]["text"] = label
    filtered = filter_semantic_cycles(G)
    assert len(filtered.edges()) == len(G.edges())


def test_filter_semantic_cycles_max_len():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
    for n in G.nodes:
        G.nodes[n]["text"] = "the"
    filtered = filter_semantic_cycles(G, max_len=3)
    assert len(filtered.edges()) == len(G.edges())


def test_entropy_triangle_threshold_simple():
    G = nx.Graph()
    G.add_edges_from(
        [(1, 2, {"weight": 1}), (2, 3, {"weight": 1}), (3, 1, {"weight": 1})]
    )
    # uniform weights -> entropy=log2(3) ~1.58 => round(5*1.58)=8
    assert entropy_triangle_threshold(G, scale=5) == 8


def test_entropy_triangle_threshold_empty_graph():
    G = nx.Graph()
    assert entropy_triangle_threshold(G) == 1


def test_entropy_triangle_threshold_missing_weight():
    G = nx.Graph()
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    thresh = entropy_triangle_threshold(G, base=np.e, scale=3)
    assert thresh >= 1
