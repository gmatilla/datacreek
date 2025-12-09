import networkx as nx

from datacreek.analysis.tpl import _diagram, sinkhorn_w1


def test_sinkhorn_zero_distance():
    g = nx.path_graph(3)
    d = _diagram(g, 1)
    assert sinkhorn_w1(d, d) == 0.0
