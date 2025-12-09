"""Persistence diagram vectorization utilities."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def persistence_image(
    diag: np.ndarray,
    sigma: float,
    grid: Tuple[np.ndarray, np.ndarray],
    *,
    weight: callable | None = None,
) -> np.ndarray:
    """Return persistence image of ``diag`` on ``grid``.

    Parameters
    ----------
    diag:
        ``(n, 2)`` array of birth-death pairs.
    sigma:
        Standard deviation of the Gaussian kernel.
    grid:
        Tuple of ``x`` and ``y`` coordinate arrays defining the image grid.
    weight:
        Optional weighting function ``w(p)`` where ``p = death - birth``.

    Returns
    -------
    np.ndarray
        Persistence image of shape ``(len(x), len(y))``.
    """

    if diag.size == 0:
        x, y = grid
        return np.zeros((len(x), len(y)), dtype=float)

    x, y = grid
    X, Y = np.meshgrid(x, y, indexing="ij")
    img = np.zeros_like(X, dtype=float)
    for b, d in diag:
        p = d - b
        w = weight(p) if weight is not None else p
        img += w * np.exp(-((X - b) ** 2 + (Y - d) ** 2) / (2 * sigma**2))
    return img


def persistence_landscape(
    diag: np.ndarray,
    t: Iterable[float],
    k_max: int,
) -> np.ndarray:
    """Return persistence landscape samples for ``diag``.

    Parameters
    ----------
    diag:
        ``(n, 2)`` array of birth-death pairs.
    t:
        Iterable of sample points.
    k_max:
        Number of landscape levels to compute.

    Returns
    -------
    np.ndarray
        Array of shape ``(k_max, len(t))`` with landscape values.
    """

    diag = np.asarray(diag, dtype=float)
    ts = np.asarray(list(t), dtype=float)
    H = np.zeros((len(diag), len(ts)), dtype=float)
    for i, (b, d) in enumerate(diag):
        H[i] = np.maximum(0.0, np.minimum(ts - b, d - ts))
    H.sort(axis=0)
    H = H[::-1]
    k = min(k_max, H.shape[0])
    landscapes = H[:k]
    if landscapes.shape[0] < k_max:
        pad = np.zeros((k_max - landscapes.shape[0], len(ts)), dtype=float)
        landscapes = np.vstack([landscapes, pad])
    return landscapes


def persistence_silhouette(
    diag: np.ndarray,
    t: Iterable[float],
    p: float = 1.0,
) -> np.ndarray:
    """Return the persistence silhouette of ``diag`` sampled on ``t``.

    The silhouette is a weighted average of triangular functions centered on
    the birth and death times of the diagram's points. A power ``p`` controls
    the weight ``w_i = (d_i - b_i)^p`` assigned to each triangle.

    Parameters
    ----------
    diag:
        ``(n, 2)`` array of birth-death pairs.
    t:
        Iterable of sample points where the silhouette is evaluated.
    p:
        Exponent applied to persistence values for weighting.

    Returns
    -------
    np.ndarray
        Array of shape ``(len(t),)`` with silhouette values.
    """

    diag = np.asarray(diag, dtype=float)
    ts = np.asarray(list(t), dtype=float)
    if diag.size == 0:
        return np.zeros_like(ts)

    weights = np.diff(diag, axis=1).ravel() ** p
    total = np.sum(weights)
    L = np.zeros_like(ts)
    for (b, d), w in zip(diag, weights):
        h = np.maximum(0.0, np.minimum(ts - b, d - ts))
        L += w * h
    if total > 0:
        L /= total
    return L


def betti_curve(diag: np.ndarray, t: Iterable[float]) -> np.ndarray:
    """Return the Betti curve of ``diag`` evaluated on ``t``.

    The Betti curve counts, for each filtration value ``t``, the number of
    intervals ``[b_i, d_i)`` in the persistence diagram that are active. It
    provides a simple vector summary of topological features across scales.

    Parameters
    ----------
    diag:
        ``(n, 2)`` array of birth-death pairs.
    t:
        Iterable of filtration values where the Betti numbers are computed.

    Returns
    -------
    np.ndarray
        Array of shape ``(len(t),)`` containing Betti numbers.
    """

    diag = np.asarray(diag, dtype=float)
    ts = np.asarray(list(t), dtype=float)
    if diag.size == 0:
        return np.zeros_like(ts)

    curve = np.zeros_like(ts, dtype=float)
    for b, d in diag:
        mask = (ts >= b) & (ts < d)
        curve[mask] += 1.0
    return curve


def diagram_entropy(diag: np.ndarray) -> float:
    r"""Return the normalized persistence entropy of ``diag``.

    The entropy is computed from the persistence lengths ``p_i = d_i - b_i``.
    After normalising ``p_i`` into probabilities, the Shannon entropy
    ``-\sum p_i \log p_i`` is returned. Empty diagrams yield ``0.0``.

    Parameters
    ----------
    diag:
        ``(n, 2)`` array of birth-death pairs.

    Returns
    -------
    float
        Normalised entropy of persistence lengths.
    """

    diag = np.asarray(diag, dtype=float)
    if diag.size == 0:
        return 0.0
    pers = np.diff(diag, axis=1).ravel()
    pers = pers[pers > 0]
    if pers.size == 0:
        return 0.0
    probs = pers / pers.sum()
    return float(-np.sum(probs * np.log(probs)))


def augment_embeddings_with_persistence(
    graph: "nx.Graph",
    embeddings: dict,
    *,
    method: str = "image",
    sigma: float = 1.0,
    grid_size: int = 8,
    t_samples: int = 32,
    k_max: int = 5,
    p: float = 1.0,
) -> dict:
    """Return embeddings concatenated with persistence features for H0/H1.

    Parameters
    ----------
    graph:
        Input graph whose persistence diagrams are computed.
    embeddings:
        Mapping of nodes to base embedding vectors.
    method:
        Which vectorisation to use: ``"image"``, ``"landscape"``,
        ``"silhouette"``, ``"betti"`` ou ``"entropy"``.
    sigma:
        Gaussian kernel bandwidth for the persistence image when
        ``method="image"``.
    grid_size:
        Resolution of the square image grid or the number of samples for
        one-dimensional summaries.
    t_samples:
        Number of sample points for landscapes or silhouettes.
    k_max:
        Number of landscape levels when ``method="landscape"``.
    p:
        Exponent for silhouette weighting when ``method="silhouette"``.

    Returns
    -------
    dict
        Mapping of node to augmented vector ``[Phi ; vec_H0 ; vec_H1]`` where
        ``vec`` depends on ``method``.
    """
    import networkx as nx

    from .fractal import persistence_diagrams

    diags = persistence_diagrams(graph, max_dim=1)
    d0 = diags.get(0, np.empty((0, 2)))
    d1 = diags.get(1, np.empty((0, 2)))
    if method not in {"image", "landscape", "silhouette", "betti", "entropy"}:
        raise ValueError("unknown method")

    if d0.size + d1.size == 0:
        mins, maxs = 0.0, 1.0
    else:
        mins = float(np.min(np.vstack([d0, d1])[:, 0]))
        maxs = float(np.max(np.vstack([d0, d1])[:, 1]))

    if method == "image":
        grid = (
            np.linspace(mins, maxs, grid_size),
            np.linspace(mins, maxs, grid_size),
        )
        v0 = persistence_image(d0, sigma, grid).ravel()
        v1 = persistence_image(d1, sigma, grid).ravel()
    else:
        ts = np.linspace(mins, maxs, t_samples)
        if method == "landscape":
            v0 = persistence_landscape(d0, ts, k_max).ravel()
            v1 = persistence_landscape(d1, ts, k_max).ravel()
        elif method == "silhouette":
            v0 = persistence_silhouette(d0, ts, p).ravel()
            v1 = persistence_silhouette(d1, ts, p).ravel()
        elif method == "betti":
            v0 = betti_curve(d0, ts).ravel()
            v1 = betti_curve(d1, ts).ravel()
        else:  # entropy
            v0 = np.array([diagram_entropy(d0)])
            v1 = np.array([diagram_entropy(d1)])

    aug = {}
    for node, vec in embeddings.items():
        base = np.asarray(list(vec), dtype=float)
        aug[node] = np.concatenate([base, v0, v1])
    return aug


def reduce_pca(X_PI: np.ndarray, n: int = 50, batch_size: int = 256) -> np.ndarray:
    """Reduce persistence image vectors using incremental PCA.

    This routine combats the curse of dimensionality by projecting the matrix
    of persistence images ``X_PI`` onto the first ``n`` principal components of
    its covariance. The incremental algorithm approximates the solution of
    ``Y = (X - \mu) W`` where the columns of ``W`` are the leading eigenvectors
    of ``(X - \mu)^\top (X - \mu)`` computed in mini-batches of size
    ``batch_size``.

    Parameters
    ----------
    X_PI:
        ``(m, d)`` array of flattened persistence images.
    n:
        Number of principal components to retain.
    batch_size:
        Size of the chunks processed per ``partial_fit`` iteration. Setting to
        ``None`` or a non-positive value fits the model in one step.

    Returns
    -------
    np.ndarray
        Matrix of shape ``(m, n)`` with the reduced representation.
    """

    try:  # prefer scikit-learn's optimised implementation
        from sklearn.decomposition import IncrementalPCA
    except Exception:  # fallback to a simple SVD-based reducer
        IncrementalPCA = None  # type: ignore

    X = np.asarray(X_PI, dtype=float)

    if IncrementalPCA is None:  # pragma: no cover - exercised when sklearn absent
        Xc = X - X.mean(axis=0, keepdims=True)
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        return U[:, :n] * S[:n]

    ipca = IncrementalPCA(n_components=n)
    if not batch_size or batch_size <= 0 or batch_size >= X.shape[0]:
        ipca.fit(X)
    else:
        for start in range(0, X.shape[0], batch_size):
            ipca.partial_fit(X[start : start + batch_size])
    return ipca.transform(X)
