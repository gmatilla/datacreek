import sys
import types

import networkx as nx
import numpy as np
import pytest

import datacreek.analysis.sheaf as sheaf


def test_sheaf_laplacian_and_incidence():
    G = nx.Graph()
    G.add_edge(0, 1, sheaf_sign=1)
    G.add_edge(1, 2, sheaf_sign=-1)
    L = sheaf.sheaf_laplacian(G)
    B = sheaf.sheaf_incidence_matrix(G)
    assert np.allclose(np.diag(L), [1, 2, 1])
    assert L[0, 1] == -1 and L[1, 2] == 1
    assert B.shape == (3, 2)
    assert np.array_equal(B[:, 0], [1, -1, 0])


def test_convolution_and_network():
    G = nx.path_graph(3)
    feats = {n: np.ones(2) * n for n in G}
    out = sheaf.sheaf_convolution(G, feats, alpha=0.5)
    assert isinstance(out, dict) and len(out) == 3
    nn = sheaf.sheaf_neural_network(G, feats, layers=2, alpha=0.1)
    for v in nn.values():
        assert (v >= 0).all()


def test_cohomology_and_consistency():
    G = nx.cycle_graph(4)
    assert sheaf.sheaf_first_cohomology(G) == 1
    final = sheaf.resolve_sheaf_obstruction(G)
    assert final == 0
    assert sheaf.sheaf_consistency_score(G) == 1.0
    batches = list(sheaf.sheaf_consistency_score_batched(G, [[0, 1, 2, 3]]))
    assert batches == [1.0]


def test_spectral_and_blocksmith(monkeypatch):
    G = nx.path_graph(3)
    # The second eigenvalue of the sheaf Laplacian for a path graph
    # exceeds 0.5 whereas the first eigenvalue is zero. Ensure that
    # spectral_bound_exceeded detects this case correctly.
    assert sheaf.spectral_bound_exceeded(G, 2, 0.5)
    monkeypatch.setitem(sys.modules, "sympy", None)
    with pytest.raises(RuntimeError):
        sheaf.block_smith_invariants(np.eye(2, dtype=int))

    mod = types.ModuleType("sympy")

    def smith_normal_form(mat):
        return np.eye(len(mat), dtype=int), None, None

    normalforms = types.ModuleType("normalforms")
    normalforms.smith_normal_form = smith_normal_form
    matrices = types.ModuleType("matrices")
    matrices.normalforms = normalforms
    mod.matrices = matrices
    mod.Matrix = lambda x: np.array(x)
    monkeypatch.setitem(sys.modules, "sympy", mod)
    monkeypatch.setitem(sys.modules, "sympy.matrices", matrices)
    monkeypatch.setitem(sys.modules, "sympy.matrices.normalforms", normalforms)
    inv = sheaf.block_smith_invariants(np.eye(2, dtype=int), block_size=1)
    assert inv == [1, 1]

    res = sheaf.block_smith(np.eye(2, dtype=int), block_size=1, lam_thresh=0.0)
    assert res == 0


def test_validate_section():
    G = nx.path_graph(3)
    score = sheaf.validate_section(G, [0, 1, 2])
    assert score == sheaf.sheaf_consistency_score(G)


def test_additional_paths(monkeypatch):
    G = nx.path_graph(3)
    # spectral bound not exceeded for high threshold
    assert not sheaf.spectral_bound_exceeded(G, 2, 10)

    # sheaf_first_cohomology_blocksmith reads config when lam_thresh None
    monkeypatch.setattr(
        "datacreek.utils.config.load_config",
        lambda: {"sheaf": {"lam_thresh": 2.0}},
    )
    assert sheaf.sheaf_first_cohomology_blocksmith(G, block_size=10) == 0

    # block_smith_invariants handles empty matrices
    mod = types.ModuleType("sympy")
    mod.Matrix = lambda x: np.array(x)
    nf = types.ModuleType("normalforms")
    nf.smith_normal_form = lambda x: (np.array(x), None, None)
    matrices = types.ModuleType("matrices")
    matrices.normalforms = nf
    mod.matrices = matrices
    monkeypatch.setitem(sys.modules, "sympy", mod)
    monkeypatch.setitem(sys.modules, "sympy.matrices", matrices)
    monkeypatch.setitem(sys.modules, "sympy.matrices.normalforms", nf)
    assert sheaf.block_smith_invariants(np.zeros((0, 0), int)) == []
    assert sheaf.block_smith(np.zeros((0, 0), int)) == 0

    # batched consistency returns one score per batch
    scores = sheaf.sheaf_consistency_score_batched(G, [[0, 1], [1, 2]])
    assert len(scores) == 2
