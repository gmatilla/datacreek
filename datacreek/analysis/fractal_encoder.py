"""Fractal feature encoding utilities."""

from __future__ import annotations

import math
from typing import Dict, Iterable

import numpy as np


def online_pca_reduce(
    matrix: np.ndarray, *, n_components: int = 256, batch_size: int = 128
) -> np.ndarray:
    """Return PCA-reduced matrix using an incremental sketch.

    Parameters
    ----------
    matrix:
        Data matrix of shape ``(n_samples, n_features)``.
    n_components:
        Target dimensionality.
    batch_size:
        Size of each partial fit batch.
    """
    try:
        from sklearn.decomposition import IncrementalPCA
    except Exception as e:  # pragma: no cover - sklearn missing
        raise RuntimeError("sklearn is required for online_pca_reduce") from e

    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    ipca.fit(matrix)
    return ipca.transform(matrix)


def fractal_encoder(
    features: Dict[object, Iterable[float]], *, embed_dim: int = 32, seed: int = 0
) -> Dict[object, list[float]]:
    """Return dense embeddings from fractal features.

    Each node's feature vector is projected with a random weight matrix and
    passed through ``tanh``.
    """
    rng = np.random.default_rng(seed)
    sample = next(iter(features.values()), None)
    if sample is None:
        return {}
    feat_dim = len(list(sample))
    W = rng.normal(scale=1 / math.sqrt(feat_dim), size=(feat_dim, embed_dim))
    encoded: Dict[object, list[float]] = {}
    for node, vec in features.items():
        x = np.asarray(list(vec), dtype=float)
        encoded[node] = np.tanh(x @ W).astype(float).tolist()
    return encoded
