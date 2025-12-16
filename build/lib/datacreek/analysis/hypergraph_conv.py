"""Spectral convolutions on hypergraphs."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Tuple

import numpy as np

# Cache for spectral radius keyed by a graph identifier. Each entry stores a
# tuple ``(lambda_max, num_edges)`` so that we can cheaply reuse the previous
# power-iteration result as long as the number of edges has not varied too much.
lambda_max_cache: Dict[str, Tuple[float, int]] = {}


def hypergraph_laplacian(B: np.ndarray, w: Iterable[float] | None = None) -> np.ndarray:
    """Return the normalized hypergraph Laplacian.

    Parameters
    ----------
    B:
        Incidence matrix of shape ``(num_nodes, num_edges)`` where ``B[v, e] = 1``
        if vertex ``v`` is incident to hyperedge ``e``.
    w:
        Optional iterable of edge weights. Defaults to ``1`` for all edges.

    Returns
    -------
    np.ndarray
        Normalized Laplacian matrix ``Delta``.
    """
    B = np.asarray(B, dtype=float)
    num_nodes, num_edges = B.shape
    if w is None:
        w = np.ones(num_edges, dtype=float)
    w = np.asarray(list(w), dtype=float)
    if w.shape[0] != num_edges:
        raise ValueError("weight length must match number of edges")

    dv = B @ w
    de = B.sum(axis=0)

    dv_inv_sqrt = np.diag(1.0 / np.sqrt(dv + 1e-12))
    de_inv = np.diag(1.0 / (de + 1e-12))
    W = np.diag(w)

    return np.eye(num_nodes) - dv_inv_sqrt @ B @ W @ de_inv @ B.T @ dv_inv_sqrt


def estimate_lambda_max(
    Delta: np.ndarray,
    g_id: str | None = None,
    num_edges: int | None = None,
    it: int = 3,
) -> float:
    r"""Estimate the largest eigenvalue :math:`\lambda_{\max}` of ``Delta``.

    A lightweight power iteration is used to avoid the costly dense
    eigendecomposition required by :func:`numpy.linalg.eigvalsh`. Starting from a
    random vector ``v`` the iteration ``v_{k+1} = \Delta v_k / \|\Delta v_k\|``
    converges to the dominant eigenvector. The corresponding Rayleigh quotient
    provides an approximation of ``\lambda_{\max}``.

    Parameters
    ----------
    Delta:
        Symmetric hypergraph Laplacian ``\Delta``.
    g_id:
        Optional graph identifier used for caching. When provided alongside
        ``num_edges``, the function will reuse the cached value if the edge
        count has not varied by more than five percent.
    num_edges:
        Number of hyperedges of the associated graph. Used to decide when the
        cache entry should be invalidated.
    it:
        Number of power iterations. A small value (``it=3``) already yields a
        good estimate in practice.

    Returns
    -------
    float
        Estimated largest eigenvalue ``\lambda_{\max}``.
    """
    Delta = np.asarray(Delta, dtype=float)
    if Delta.ndim != 2 or Delta.shape[0] != Delta.shape[1]:
        raise ValueError("Delta must be a square matrix")

    if g_id is not None and num_edges is not None:
        cached = lambda_max_cache.get(g_id)
        if cached is not None:
            lamb, prev_edges = cached
            if prev_edges > 0 and abs(num_edges - prev_edges) / prev_edges <= 0.05:
                return lamb

    # Start from a random but deterministic vector for reproducibility.
    rng = np.random.default_rng(0)
    v = rng.standard_normal(Delta.shape[0])
    v /= np.linalg.norm(v) + 1e-12
    for _ in range(it):
        v = Delta @ v
        norm = np.linalg.norm(v)
        if norm == 0.0:
            lamb = 0.0
            break
        v /= norm
    else:
        lamb = float(v @ (Delta @ v))

    if g_id is not None and num_edges is not None:
        lambda_max_cache[g_id] = (lamb, num_edges)
    return lamb


logger = logging.getLogger(__name__)


def chebyshev_conv(
    X: np.ndarray,
    Delta: np.ndarray,
    K: int | None = None,
    theta: Iterable[float] | None = None,
    g_id: str | None = None,
    num_edges: int | None = None,
    lambda_k: float | None = None,
) -> np.ndarray:
    """Return Chebyshev spectral convolution on hypergraph features.

    Parameters
    ----------
    X:
        Node feature matrix of shape ``(num_nodes, feat_dim)``.
    Delta:
        Normalized hypergraph Laplacian.
    K:
        Order of the Chebyshev approximation. If ``None`` the value is chosen
        adaptively from the spectral gap ``Δλ`` using the formula

        .. math::

            K = \lceil \pi / \arccos(1 - Δλ) \rceil.

    theta:
        Iterable of ``K`` filter coefficients. Defaults to ones.
    g_id:
        Optional graph identifier used to cache ``λ_max`` estimations.
    num_edges:
        Number of hyperedges of the graph, required when ``g_id`` is provided.
    lambda_k:
        Target eigenvalue ``λ_K`` used to compute the spectral gap
        ``Δλ = λ_max - λ_K`` when ``K`` is not supplied. Defaults to ``0.9`` of
        ``λ_max`` if omitted.

    Returns
    -------
    np.ndarray
        Filtered features with the same shape as ``X``.
    """
    X = np.asarray(X, dtype=float)
    Delta = np.asarray(Delta, dtype=float)

    lamb_max = estimate_lambda_max(Delta, g_id=g_id, num_edges=num_edges)
    if lamb_max == 0.0:
        lamb_max = 1.0

    if K is None:
        if lambda_k is None:
            lambda_k = 0.9 * lamb_max
        delta_lambda = lamb_max - float(lambda_k)
        K = int(np.ceil(np.pi / np.arccos(1 - delta_lambda)))
    logger.info("spec_K_chosen=%d", K)

    if theta is None:
        theta = np.ones(K, dtype=float)
    theta = np.asarray(list(theta), dtype=float)
    if theta.shape[0] != K:
        raise ValueError("theta length must equal K")

    Delta_tilde = (2.0 / lamb_max) * Delta - np.eye(Delta.shape[0])

    T_k_minus = X
    out = theta[0] * T_k_minus
    if K > 1:
        T_k = Delta_tilde @ X
        out = out + theta[1] * T_k
    else:
        return out
    for k in range(2, K):
        T_k_plus = 2 * Delta_tilde @ T_k - T_k_minus
        out = out + theta[k] * T_k_plus
        T_k_minus, T_k = T_k, T_k_plus
    return out
