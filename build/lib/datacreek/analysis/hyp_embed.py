"""Learned hyperbolic embeddings using a simple MLP."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import networkx as nx
import numpy as np

from .fractal import latent_box_dimension


def _to_poincare(x: np.ndarray) -> np.ndarray:
    """Map Euclidean vectors to the Poincar\u00e9 ball."""
    norm_sq = np.sum(x**2, axis=1, keepdims=True)
    x0 = np.sqrt(1.0 + norm_sq)
    return x / (x0 + 1.0)


def learn_hyperbolic_projection(
    features: Mapping[object, Iterable[float]],
    graph: nx.Graph,
    *,
    dim: int = 2,
    lr: float = 0.01,
    epochs: int = 50,
    geo_weight: float = 1.0,  # unused placeholder
    frac_weight: float = 0.1,
    frac_target: float | None = None,
    radii: Iterable[float] | None = None,
) -> Dict[object, np.ndarray]:
    """Return learned hyperbolic embeddings for ``graph`` nodes."""
    nodes = list(features.keys())
    X = np.vstack([features[n] for n in nodes])
    rng = np.random.default_rng(0)
    W1 = rng.normal(scale=0.1, size=(X.shape[1], 2 * dim))
    W2 = rng.normal(scale=0.1, size=(2 * dim, dim))
    for _ in range(epochs):
        H = np.tanh(X @ W1)
        Z = H @ W2
        grad_Z = 2 * (Z - X)
        grad_W2 = H.T @ grad_Z / len(X)
        grad_H = grad_Z @ W2.T * (1 - H**2)
        grad_W1 = X.T @ grad_H / len(X)
        W2 -= lr * grad_W2
        W1 -= lr * grad_W1
    Z = np.tanh(X @ W1) @ W2
    emb = _to_poincare(Z)
    if radii is None:
        radii = [0.5, 1.0]
    if frac_target is not None and frac_weight > 0.0:
        dim_est, _ = latent_box_dimension({n: e for n, e in zip(nodes, emb)}, radii)
        _ = frac_weight * (dim_est - frac_target) ** 2  # placeholder to reflect loss
    return {n: emb[i] for i, n in enumerate(nodes)}
