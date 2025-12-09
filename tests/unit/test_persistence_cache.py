import networkx as nx
import numpy as np
import pytest

from datacreek.analysis.fractal import (
    _persistence_diagrams_cached,
    gd,
    persistence_diagrams,
)

if gd is None:
    pytest.skip("gudhi not available", allow_module_level=True)


def test_persistence_diagrams_cache_hits() -> None:
    _persistence_diagrams_cached.cache_clear()
    g = nx.path_graph(4)
    diags1 = persistence_diagrams(g, max_dim=1)
    hits_before = _persistence_diagrams_cached.cache_info().hits
    diags2 = persistence_diagrams(g, max_dim=1)
    hits_after = _persistence_diagrams_cached.cache_info().hits
    assert hits_after == hits_before + 1
    for dim in diags1:
        assert np.array_equal(diags1[dim], diags2[dim])
