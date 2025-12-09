import importlib
import types

import numpy as np
import pytest

import datacreek.analysis.chebyshev_diag as cd


@pytest.mark.heavy
def test_chebyshev_diag_fallback(monkeypatch):
    """Fallback path should compute Hutchinson estimate."""
    # ensure scipy parts missing
    monkeypatch.setattr(cd, "la", None, raising=False)
    monkeypatch.setattr(cd, "sp", None, raising=False)
    monkeypatch.setattr(cd, "eigsh", None, raising=False)
    monkeypatch.setattr(cd, "iv", None, raising=False)
    L = np.eye(2)
    res = cd.chebyshev_diag_hutchpp(
        L, 1.0, order=3, samples=4, rng=np.random.default_rng(0)
    )
    assert res.shape == (2,)
    # fallback degenerates to diagonal of L
    assert np.allclose(res, [1.0, 1.0])


@pytest.mark.heavy
def test_chebyshev_diag_scipy_stub(monkeypatch):
    """Stub scipy dependencies to exercise the main routine."""

    # create simple stubs mimicking scipy interfaces
    class la_stub:
        eigvalsh = staticmethod(np.linalg.eigvalsh)

    class sp_stub:
        @staticmethod
        def issparse(x):
            return False

        @staticmethod
        def identity(n, format="csc"):
            return np.eye(n)

    def eigsh_stub(L, k=1, which="LM", return_eigenvectors=False):
        return np.array([np.linalg.eigvalsh(L)[-1]])

    iv_stub = lambda arr, x: np.ones_like(arr, dtype=float)

    monkeypatch.setattr(cd, "la", la_stub, raising=False)
    monkeypatch.setattr(cd, "sp", sp_stub, raising=False)
    monkeypatch.setattr(cd, "eigsh", eigsh_stub, raising=False)
    monkeypatch.setattr(cd, "iv", iv_stub, raising=False)

    L = np.diag([1.0, 2.0])
    res = cd.chebyshev_diag_hutchpp(
        L, 0.1, order=3, samples=4, rng=np.random.default_rng(0)
    )
    assert res.shape == (2,)
    assert np.all(np.isfinite(res))
