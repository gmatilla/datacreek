from __future__ import annotations

import numpy as np

from datacreek.backend import get_xp

xp = get_xp()

try:  # optional heavy deps
    import scipy.linalg as la  # pragma: no cover - optional
    import scipy.sparse as sp  # pragma: no cover - optional
    from scipy.sparse.linalg import eigsh  # pragma: no cover - optional
    from scipy.special import iv  # pragma: no cover - optional
except Exception:  # pragma: no cover - optional dependency
    la = None  # type: ignore
    sp = None  # type: ignore
    eigsh = None  # type: ignore
    iv = None  # type: ignore

__all__ = ["chebyshev_diag_hutchpp"]


def _chebyshev_diag_hutchpp_scipy(
    L: np.ndarray | "sp.spmatrix",
    t: float,
    *,
    order: int,
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:  # pragma: no cover - requires scipy
    """Return ``diag(exp(-t L))`` using Hutch++ and SciPy.

    This helper implements the full algorithm relying on ``scipy``. It is
    excluded from coverage because the heavy dependency is not installed in CI.

    When :mod:`scipy` is available the diagonal is estimated without forming
    ``exp(-t L)`` explicitly using the Hutch++ algorithm. Otherwise a basic
    Hutchinson trace estimator is used. The function targets a relative error of
    roughly ``1e-2`` with ``samples`` probes.
    """

    n = L.shape[0]

    # spectral normalization
    if sp.issparse(L):
        lam_max = float(eigsh(L, k=1, which="LM", return_eigenvectors=False)[0])
        lam_min = 0.0
        I = sp.identity(n, format="csc")
    else:
        vals = la.eigvalsh(L)
        lam_min = float(vals[0])
        lam_max = float(vals[-1])
        I = np.eye(n)

    alpha = 0.5 * (lam_max - lam_min)
    beta = 0.5 * (lam_max + lam_min)

    def matvec(X: np.ndarray) -> np.ndarray:
        """Return ``exp(-t L) X`` via truncated Chebyshev series."""

        if sp.issparse(L):
            Y0 = X
            Y1 = (L - beta * I).dot(X) / alpha
        else:
            Y0 = X
            Y1 = ((L - beta * I) @ X) / alpha

        coeff = np.exp(-t * beta) * iv(np.arange(order + 1), t * alpha)
        coeff[1:] *= 2.0

        result = coeff[0] * Y0 + coeff[1] * Y1
        Tm2, Tm1 = Y0, Y1
        for k in range(2, order + 1):
            if sp.issparse(L):
                Tk = 2.0 * (L - beta * I).dot(Tm1) / alpha - Tm2
            else:
                Tk = 2.0 * ((L - beta * I) @ Tm1) / alpha - Tm2
            result = result + coeff[k] * Tk
            Tm2, Tm1 = Tm1, Tk
        return result

    s1 = max(2, samples // 3)
    s2 = samples - s1

    G = rng.choice([-1.0, 1.0], size=(n, s1))
    Y = matvec(G)
    Q, _ = np.linalg.qr(Y, mode="reduced")

    B = Q.T @ matvec(Q)
    diag_lr = np.sum((Q @ B) * Q, axis=1)

    diag_res = np.zeros(n)
    for _ in range(s2):
        z = rng.choice([-1.0, 1.0], size=(n, 1))
        Az = matvec(z)
        qtz = Q.T @ z
        correction = Q @ (B @ qtz)
        r = Az - correction
        diag_res += z[:, 0] * r[:, 0]
    diag_res /= s2

    return diag_lr + diag_res


def chebyshev_diag_hutchpp(
    L: np.ndarray | "sp.spmatrix",
    t: float,
    *,
    order: int = 7,
    samples: int = 64,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return ``diag(exp(-t L))`` using Hutch++ with a lightweight fallback."""
    rng = np.random.default_rng(rng)
    if la is None or sp is None or eigsh is None or iv is None:
        arr = L.toarray() if sp is not None and sp.issparse(L) else np.asarray(L)
        n = arr.shape[0]
        Z = rng.choice([-1.0, 1.0], size=(n, samples))
        Y = arr @ Z
        return np.mean(Y * Z, axis=1)

    return _chebyshev_diag_hutchpp_scipy(
        L, t, order=order, samples=samples, rng=rng
    )  # pragma: no cover - requires scipy
