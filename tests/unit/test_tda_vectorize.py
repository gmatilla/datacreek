import numpy as np
import pytest

from datacreek.analysis import reduce_pca


def test_reduce_pca_deterministic_and_matches_sklearn():
    """Shape is reduced and, if available, matches scikit-learn's IPCA."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 20))
    Y1 = reduce_pca(X, n=5, batch_size=10)
    Y2 = reduce_pca(X, n=5, batch_size=10)
    assert Y1.shape == (40, 5)
    np.testing.assert_allclose(Y1, Y2)

    try:
        from sklearn.decomposition import IncrementalPCA
    except Exception:
        pytest.skip("scikit-learn not available")
    ipca = IncrementalPCA(n_components=5)
    for start in range(0, X.shape[0], 10):
        ipca.partial_fit(X[start : start + 10])
    Y_ref = ipca.transform(X)
    np.testing.assert_allclose(Y1, Y_ref, rtol=1e-5, atol=1e-8)
