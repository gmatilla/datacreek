import sys
import types

import networkx as nx
import numpy as np
import pytest

from datacreek.analysis import sheaf


@pytest.mark.heavy
def test_laplacian_and_incidence():
    g = nx.Graph()
    g.add_edge("a", "b", sheaf_sign=1)
    g.add_edge("b", "c", sheaf_sign=-1)

    L = sheaf.sheaf_laplacian(g)
    B = sheaf.sheaf_incidence_matrix(g)

    assert L.shape == (3, 3)
    # expected signed Laplacian for path a-b-c with one negative sign
    expected_L = np.array([[1, -1, 0], [-1, 2, 1], [0, 1, 1]], dtype=float)
    assert np.allclose(L, expected_L)

    expected_B = np.array([[1, 0], [-1, 1], [0, 1]])
    assert np.array_equal(B, expected_B)


@pytest.mark.heavy
def test_convolution_and_network():
    g = nx.path_graph(["a", "b", "c"])
    feats = {
        "a": np.array([1.0, 0.0]),
        "b": np.array([0.0, 1.0]),
        "c": np.array([1.0, 1.0]),
    }
    out = sheaf.sheaf_convolution(g, feats, alpha=0.5)
    assert out["a"][0] < feats["a"][0]
    net = sheaf.sheaf_neural_network(g, feats, layers=2, alpha=0.1)
    assert len(net) == 3 and all(n in net for n in g)


@pytest.mark.heavy
def test_cohomology_and_resolution():
    g = nx.cycle_graph(3)
    before = sheaf.sheaf_first_cohomology(g)
    assert before > 0
    after = sheaf.resolve_sheaf_obstruction(g)
    assert after <= before
    assert sheaf.sheaf_first_cohomology(g) == after


@pytest.mark.heavy
def test_scores_and_batches():
    g = nx.cycle_graph(3)
    base = sheaf.sheaf_consistency_score(g)
    sheaf.resolve_sheaf_obstruction(g)
    improved = sheaf.sheaf_consistency_score(g)
    assert improved >= base
    scores = sheaf.sheaf_consistency_score_batched(g, [[0, 1], [1, 2]])
    assert len(scores) == 2 and all(0 <= s <= 1 for s in scores)
    assert scores[0] == scores[1]


@pytest.mark.heavy
def test_spectral_bound_and_blocks(monkeypatch):
    g = nx.path_graph(3)
    # second eigenvalue of standard Laplacian exceeds 0.1
    assert sheaf.spectral_bound_exceeded(g, 2, 0.1)
    monkeypatch.setitem(sys.modules, "scipy", types.SimpleNamespace())
    assert not sheaf.spectral_bound_exceeded(g, 5, 100.0)


@pytest.mark.heavy
def test_blocksmith_functions(monkeypatch):
    mod = types.ModuleType("sympy")
    mod.Matrix = lambda x: np.array(x)
    nf = types.ModuleType("sympy.matrices.normalforms")
    nf.smith_normal_form = lambda m: (np.array(m), None, None)
    matrices = types.ModuleType("sympy.matrices")
    matrices.normalforms = nf
    mod.matrices = matrices
    sys.modules["sympy"] = mod
    sys.modules["sympy.matrices"] = matrices
    sys.modules["sympy.matrices.normalforms"] = nf

    delta = np.array([[1, 0], [0, 2]])
    inv = sheaf.block_smith_invariants(delta, block_size=1)
    assert inv == [1]

    from datacreek.utils import config

    monkeypatch.setattr(config, "load_config", lambda: {"sheaf": {"lam_thresh": 0.5}})
    rank = sheaf.block_smith(delta, block_size=1, lam_thresh=None)
    assert rank in {0, 1}

    monkeypatch.setattr(sheaf, "block_smith", lambda *a, **k: 1)
    g = nx.path_graph(2)
    assert sheaf.sheaf_first_cohomology_blocksmith(g, block_size=2) == 0

    assert sheaf.validate_section(g, [0, 1]) >= 0
