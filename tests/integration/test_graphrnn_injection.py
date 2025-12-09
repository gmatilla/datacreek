import networkx as nx

from datacreek.analysis.fractal import inject_and_validate


def test_inject_and_validate():
    g = nx.path_graph(4)
    score = inject_and_validate(g, 2, 1)
    assert 0.0 <= score <= 1.0
