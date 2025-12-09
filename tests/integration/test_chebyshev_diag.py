import networkx as nx
import numpy as np
import pytest

from datacreek.analysis.chebyshev_diag import chebyshev_diag_hutchpp


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("scipy") is None, reason="scipy required"
)
def test_chebyshev_diag_accuracy():
    import scipy.linalg as la

    g = nx.path_graph(6)
    L = nx.normalized_laplacian_matrix(g).toarray()
    t = 0.5
    exact = np.diag(la.expm(-t * L))
    approx = chebyshev_diag_hutchpp(L, t, order=5, samples=64, rng=0)
    rel_err = np.abs(approx - exact) / exact
    assert np.mean(rel_err) < 0.01


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("scipy") is None, reason="scipy required"
)
@pytest.mark.heavy
def test_chebyshev_diag_variance():
    import scipy.linalg as la

    g = nx.path_graph(6)
    L = nx.normalized_laplacian_matrix(g).toarray()
    t = 0.5
    exact = np.diag(la.expm(-t * L))

    runs = []
    for i in range(100):
        runs.append(chebyshev_diag_hutchpp(L, t, order=5, samples=64, rng=i))
    arr = np.stack(runs)
    mape = np.mean(np.abs(arr.mean(axis=0) - exact) / exact)
    var = arr.var(axis=0).max()
    assert mape < 0.01
    assert var < 1e-3
