import networkx as nx
import pytest

from datacreek.analysis.sheaf_hyper_bridge import (
    sheaf_hyper_bridge_score,
    top_k_incoherent,
)


def test_sheaf_hyper_bridge_score_dense():
    g = nx.complete_graph(5)
    for u, v in g.edges():
        g[u][v]["sheaf_sign"] = 1
    score = sheaf_hyper_bridge_score(g)
    assert 0.0 <= score <= 1.0
    assert score > 0.7


def test_top_k_incoherent_edges():
    g = nx.Graph()
    g.add_edge(0, 1, sheaf_sign=1)
    g.add_edge(1, 2, sheaf_sign=2)  # different sign to induce mismatch
    out = top_k_incoherent(g, k=2, tau=0.1)
    assert out == [((1, 2), pytest.approx(3.0))]
    # High threshold filters the edge
    assert top_k_incoherent(g, k=2, tau=10.0) == []
