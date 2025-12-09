import numpy as np
import pytest

import datacreek.analysis.chebyshev_diag as cd

try:  # optional heavy dependency for spectral ops
    import scipy.sparse.linalg  # type: ignore

    _has_scipy = True
except Exception:  # pragma: no cover - dependency missing
    _has_scipy = False


def test_chebyshev_diag_fallback(monkeypatch):
    """Fallback path uses Hutchinson estimator without scipy."""
    monkeypatch.setattr(cd, "la", None)
    monkeypatch.setattr(cd, "sp", None)
    monkeypatch.setattr(cd, "eigsh", None)
    monkeypatch.setattr(cd, "iv", None)
    L = np.diag([1.0, 2.0])
    diag = cd.chebyshev_diag_hutchpp(
        L, t=0.5, order=3, samples=10, rng=np.random.default_rng(0)
    )
    assert np.allclose(diag, np.array([1.0, 2.0]))


def test_chebyshev_diag_scipy():
    """When scipy is present the estimator approximates exp(-t L)."""
    if not _has_scipy:
        pytest.skip("requires scipy")
    L = np.diag([0.0, 2.0])
    diag = cd.chebyshev_diag_hutchpp(
        L, t=0.1, order=5, samples=50, rng=np.random.default_rng(1)
    )
    expected = np.array([1.0, np.exp(-0.2)])
    assert sorted(np.round(diag, 4)) == pytest.approx(
        sorted(np.round(expected, 4)), rel=1e-2
    )
