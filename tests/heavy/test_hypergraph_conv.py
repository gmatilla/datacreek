import numpy as np
import pytest

from datacreek.analysis.hypergraph_conv import chebyshev_conv, hypergraph_laplacian


@pytest.mark.heavy
def test_hypergraph_laplacian_psd():
    B = np.array([[1, 0], [1, 1], [0, 1]])
    L = hypergraph_laplacian(B)
    evals, evecs = np.linalg.eigh(L)
    assert np.all(evals >= -1e-8)
    diff = np.linalg.norm(evecs.T @ evecs - np.eye(B.shape[0]))
    assert diff < 1e-6


@pytest.mark.heavy
def test_chebyshev_conv_shape():
    B = np.array([[1, 0], [1, 1], [0, 1]])
    L = hypergraph_laplacian(B)
    X = np.eye(3)
    out = chebyshev_conv(X, L, K=3)
    assert out.shape == X.shape
    assert np.all(np.isfinite(out))
