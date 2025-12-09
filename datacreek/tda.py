"""Topological data analysis helpers.

This module implements utilities used to enrich embeddings with a persistent
homology sketch and to choose an optimal UMAP lens based on the
trustworthiness metric.  Both helpers are lightweight wrappers around scientific
libraries and primarily exist to centralise small pieces of mathematical logic.
"""

from __future__ import annotations

import hashlib
from typing import Dict

import numpy as np

try:  # GUDHI is optional at runtime but required for persistence sketches
    import gudhi  # type: ignore
except Exception:  # pragma: no cover - dependency not installed
    gudhi = None


def persistence_minhash(points: np.ndarray, num_hashes: int = 8) -> bytes:
    """Return a 512-bit MinHash signature of the persistence diagram.

    The diagram is computed from a 2-dimensional Rips complex over ``points``
    using GUDHI's :func:`persistence` with coefficients in :math:`\mathbb{Z}_2`.
    The resulting birth/death pairs are fed through ``num_hashes`` simple hash
    functions, and the minimum value for each function forms the signature.  The
    default of ``num_hashes=8`` yields a 512-bit digest (8 × 64-bit integers).
    """

    if gudhi is None:  # pragma: no cover - would fail in production
        raise RuntimeError("gudhi library is required for persistence sketches")

    rips = gudhi.RipsComplex(points=points)
    st = rips.create_simplex_tree(max_dimension=2)
    diag = st.persistence(p=2)

    max_int = 2**64 - 1
    signature = [max_int] * num_hashes
    for birth, death in diag:
        token = f"{birth:.6f},{death:.6f}".encode()
        for i in range(num_hashes):
            h = hashlib.sha256(i.to_bytes(4, "little") + token).digest()[:8]
            val = int.from_bytes(h, "big")
            if val < signature[i]:
                signature[i] = val
    return b"".join(v.to_bytes(8, "big") for v in signature)


def select_best_lens(
    embeddings: np.ndarray,
    candidates: Dict[str, np.ndarray],
    threshold: float = 0.95,
    n_neighbors: int = 5,
) -> str:
    """Return the name of the UMAP lens whose trustworthiness is highest.

    Parameters
    ----------
    embeddings:
        Original high-dimensional data used as ground truth.
    candidates:
        Mapping from lens name to its low-dimensional embedding.
    threshold:
        Minimum acceptable trustworthiness.  If a lens exceeds this value, the
        first such lens is returned immediately.  Otherwise, the lens with the
        highest score is returned.
    n_neighbors:
        Neighborhood size used when computing trustworthiness.
    """

    def _pairwise_dist(data: np.ndarray) -> np.ndarray:
        """Return the pairwise Euclidean distances for ``data``."""

        diff = data[:, None, :] - data[None, :, :]
        return np.linalg.norm(diff, axis=2)

    def _trustworthiness(x: np.ndarray, y: np.ndarray, k: int) -> float:
        """Compute the trustworthiness metric without scikit-learn."""

        n = x.shape[0]
        k = min(k, n - 1)
        dist_x = _pairwise_dist(x)
        dist_y = _pairwise_dist(y)
        rank_x = np.argsort(np.argsort(dist_x, axis=1), axis=1)
        rank_y = np.argsort(np.argsort(dist_y, axis=1), axis=1)
        t = 0.0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if rank_y[i, j] <= k and rank_x[i, j] > k:
                    t += rank_x[i, j] - k
        normaliser = n * k * (2 * n - 3 * k - 1)
        return 1 - (2 * t) / normaliser if normaliser else 1.0

    best_name = ""
    best_score = -1.0
    for name, lens in candidates.items():
        score = _trustworthiness(embeddings, lens, n_neighbors)
        if score > threshold:
            return name
        if score > best_score:
            best_name, best_score = name, score
    return best_name
