"""Property-based tests for TDA utilities."""

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("hypothesis")
from hypothesis import given, settings
from hypothesis import strategies as st

from datacreek.analysis.fractal import box_counting_dimension, gd, persistence_diagrams

if gd is None:
    pytest.skip("gudhi not available", allow_module_level=True)


@given(st.integers(min_value=4, max_value=8))
@settings(max_examples=10)
def test_box_counting_monotonic(n: int) -> None:
    """Box counts should decrease with increasing radius."""
    g = nx.path_graph(n)
    radii = [1, 2, 3]
    _, counts = box_counting_dimension(g, radii)
    values = [c for _, c in counts]
    assert values == sorted(values, reverse=True)


@given(st.integers(min_value=3, max_value=6))
@settings(max_examples=10)
def test_persistence_diagrams_invariance(n: int) -> None:
    """Diagrams are invariant under node relabeling."""
    g = nx.path_graph(n)
    mapping = {i: j for i, j in enumerate(reversed(range(n)))}
    g2 = nx.relabel_nodes(g, mapping)
    d1 = persistence_diagrams(g, max_dim=1)
    d2 = persistence_diagrams(g2, max_dim=1)
    for dim in d1:
        assert np.allclose(np.sort(d1[dim], axis=0), np.sort(d2[dim], axis=0))
