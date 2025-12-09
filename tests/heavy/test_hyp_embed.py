import networkx as nx
import numpy as np
import pytest

from datacreek.analysis import learn_hyperbolic_projection


@pytest.mark.heavy
def test_learn_hyperbolic_projection_basic():
    g = nx.path_graph(3)
    feats = {n: np.array([float(n)]) for n in g.nodes()}
    emb = learn_hyperbolic_projection(feats, g, dim=2, epochs=5)
    assert set(emb) == set(g.nodes())
    for vec in emb.values():
        assert np.all(np.isfinite(vec)) and len(vec) == 2
