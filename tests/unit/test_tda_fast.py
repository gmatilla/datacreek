import networkx as nx
import numpy as np

from datacreek.analysis.tda_fast import _fast_diagrams_cached, fast_persistence_diagrams


def test_fast_persistence_diagrams_h0_h1():
    g = nx.cycle_graph(4)
    diags = fast_persistence_diagrams(g, max_dim=1)
    d0 = diags[0]
    d1 = diags[1]
    assert d0.shape == (3, 2)
    assert np.allclose(d0[:, 0], 0.0)
    assert d1.shape == (1, 2)
    assert np.all(d1[:, 0] <= d1[:, 1])


def test_fast_persistence_diagrams_cache():
    _fast_diagrams_cached.cache_clear()
    g = nx.path_graph(4)
    fast_persistence_diagrams(g)
    hits_before = _fast_diagrams_cached.cache_info().hits
    fast_persistence_diagrams(g)
    hits_after = _fast_diagrams_cached.cache_info().hits
    assert hits_after == hits_before + 1
