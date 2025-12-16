from __future__ import annotations

from typing import Iterable

import networkx as nx
import numpy as np

try:  # optional dependency
    import scipy.sparse as sp
except Exception:  # pragma: no cover - optional
    sp = None  # type: ignore

__all__ = ["estimate_lambda_max", "update_graphwave_bandwidth"]


def estimate_lambda_max(graph: nx.Graph, iters: int = 5) -> float:
    """Return stochastic estimate of the Laplacian spectral radius.

    Parameters
    ----------
    graph:
        Input graph.
    iters:
        Number of power iterations. ``5`` gives a rough estimate
        in ``O(|E|)`` time.

    Returns
    -------
    float
        Approximate largest eigenvalue of the normalized Laplacian.
    """

    n = graph.number_of_nodes()
    if n == 0:
        return 0.0

    if sp is not None:
        A = nx.to_scipy_sparse_array(graph, format="csr")
        deg = np.asarray(A.sum(axis=1)).ravel()
        with np.errstate(divide="ignore"):
            d_isqrt = 1.0 / np.sqrt(deg)
        d_isqrt[~np.isfinite(d_isqrt)] = 0.0
        D_isqrt = sp.diags(d_isqrt)
        L = sp.eye(n, format="csr") - D_isqrt @ A @ D_isqrt
    else:  # pragma: no cover - scipy missing
        A = nx.to_numpy_array(graph)
        deg = A.sum(axis=1)
        with np.errstate(divide="ignore"):
            d_isqrt = 1.0 / np.sqrt(deg)
        d_isqrt[~np.isfinite(d_isqrt)] = 0.0
        L = np.eye(n) - d_isqrt[:, None] * A * d_isqrt

    x = np.random.rand(n)
    x /= np.linalg.norm(x)
    for _ in range(iters):
        x = L @ x
        norm = np.linalg.norm(x)
        if norm == 0:
            break
        x /= norm
    return float(x @ (L @ x))


def update_graphwave_bandwidth(
    graph: nx.Graph, *, threshold: float = 0.05, iters: int = 5
) -> float:
    """Update diffusion scale for GraphWave based on ``lambda_max``.

    When the estimated spectral radius changes by more than ``threshold``
    relative, the value ``graph.graph['gw_t']`` is set to ``3 / lambda_max``.

    Parameters
    ----------
    graph:
        Graph on which to compute the bandwidth.
    threshold:
        Relative change required to recompute the scale.
    iters:
        Number of power iterations used for the estimation.

    Returns
    -------
    float
        Current diffusion time ``t`` stored on ``graph``.
    """

    lmax = estimate_lambda_max(graph, iters)
    prev = graph.graph.get("gw_lambda_max")
    if prev is None or abs(lmax - prev) / max(prev, 1e-12) > threshold:
        graph.graph["gw_lambda_max"] = lmax
        graph.graph["gw_t"] = 3.0 / max(lmax, 1e-12)
    return float(graph.graph.get("gw_t", 3.0 / max(lmax, 1e-12)))
