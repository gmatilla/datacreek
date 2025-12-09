"""Property-based invariants for GraphWave and Möbius operations."""

import networkx as nx
import numpy as np
import pytest

from datacreek.analysis.fractal import graphwave_embedding_chebyshev
from datacreek.analysis.poincare_recentering import _mobius_add, _mobius_neg

pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


@given(st.integers(min_value=3, max_value=10))
@settings(max_examples=30)
def test_graphwave_norm_invariance(n: int) -> None:
    """Embeddings norms should be constant on cycle graphs."""
    g = nx.cycle_graph(n)
    emb = graphwave_embedding_chebyshev(g, [0.5], num_points=4, order=3)
    norms = [np.linalg.norm(v) for v in emb.values()]
    if not np.allclose(norms, norms[0]):
        raise AssertionError


@given(st.integers(min_value=3, max_value=6))
@settings(max_examples=30)
def test_graphwave_symmetry(n: int) -> None:
    """Embeddings for path graphs are symmetric around the center."""
    g = nx.path_graph(n)
    emb = graphwave_embedding_chebyshev(g, [0.5], num_points=4, order=3)
    for i in range(n):
        j = n - 1 - i
        if not np.allclose(emb[i], emb[j]):
            raise AssertionError


vector = st.lists(st.floats(-0.4, 0.4), min_size=2, max_size=2).map(
    lambda values: np.array(values, dtype=float)
)


@given(x=vector, y=vector)
@settings(max_examples=30)
def test_mobius_add_inverse(x: np.ndarray, y: np.ndarray) -> None:
    """(x⊕y)⊖y should recover x under Möbius operations."""
    res = _mobius_add(x, y, clamp=False)
    back = _mobius_add(res, _mobius_neg(y), clamp=False)
    if not np.allclose(back, x, atol=3e-1):
        raise AssertionError
