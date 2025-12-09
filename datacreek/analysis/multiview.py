"""Multi-view embedding utilities.

This module implements simple helpers to correlate Euclidean
(Node2Vec) and spectral (GraphWave) embeddings with hyperbolic
(Poincaré) ones. Functions are intentionally lightweight so that tests
can run without heavy dependencies.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.cross_decomposition import CCA

# Base directory for cached artifacts
CACHE_ROOT = os.environ.get("DATACREEK_CACHE", "./cache")


def product_embedding(
    hyperbolic: Dict[object, Iterable[float]],
    euclidean: Dict[object, Iterable[float]],
) -> Dict[object, np.ndarray]:
    """Return concatenated product-manifold embeddings.

    Parameters
    ----------
    hyperbolic:
        Mapping of nodes to Poincaré vectors.
    euclidean:
        Mapping of nodes to Node2Vec vectors.

    Returns
    -------
    dict
        Node to combined embedding ``[x_H, x_E]`` as numpy array.
    """
    nodes = hyperbolic.keys() & euclidean.keys()
    emb: Dict[object, np.ndarray] = {}
    for n in nodes:
        h = np.asarray(hyperbolic[n], dtype=float)
        e = np.asarray(euclidean[n], dtype=float)
        emb[n] = np.concatenate([h, e])
    return emb


# --- Training on product manifold ---


def train_product_manifold(
    hyperbolic: Dict[object, Iterable[float]],
    euclidean: Dict[object, Iterable[float]],
    contexts: Iterable[tuple[object, object]],
    *,
    alpha: float = 0.5,
    lr: float = 0.01,
    epochs: int = 1,
) -> tuple[Dict[object, np.ndarray], Dict[object, np.ndarray]]:
    """Train embeddings by minimizing a simplified product-manifold loss."""
    nodes = hyperbolic.keys() & euclidean.keys()
    if not nodes:
        return {}, {}
    h = {n: np.asarray(hyperbolic[n], dtype=float) for n in nodes}
    e = {n: np.asarray(euclidean[n], dtype=float) for n in nodes}
    for _ in range(epochs):
        for u, v in contexts:
            if u not in nodes or v not in nodes:
                continue
            h_u, h_v = h[u], h[v]
            e_u, e_v = e[u], e[v]
            grad_h_u = 2 * alpha * (h_u - h_v)
            grad_h_v = -grad_h_u
            grad_e_u = 2 * (1 - alpha) * (e_u - e_v)
            grad_e_v = -grad_e_u
            h_u -= lr * grad_h_u
            h_v -= lr * grad_h_v
            e_u -= lr * grad_e_u
            e_v -= lr * grad_e_v
            for vec in (h_u, h_v):
                nrm = np.linalg.norm(vec)
                if nrm >= 1.0:
                    vec *= 0.999 / nrm
            h[u], h[v], e[u], e[v] = h_u, h_v, e_u, e_v
    return h, e


def aligned_cca(
    n2v: Dict[object, Iterable[float]],
    gw: Dict[object, Iterable[float]],
    *,
    n_components: int = 32,
) -> Tuple[Dict[object, np.ndarray], CCA]:
    """Return A-CCA latent vectors for nodes present in both mappings.

    Parameters
    ----------
    n2v:
        Node2Vec embeddings.
    gw:
        GraphWave embeddings.
    n_components:
        Output dimension of the latent space.

    Returns
    -------
    Tuple[dict, CCA]
        Mapping of node to latent vector and fitted CCA object.
    """
    nodes = n2v.keys() & gw.keys()
    if not nodes:
        return {}, CCA(n_components=n_components)
    X = np.vstack([np.asarray(n2v[n], dtype=float) for n in nodes])
    Y = np.vstack([np.asarray(gw[n], dtype=float) for n in nodes])
    cca = CCA(n_components=n_components)
    X_c, _ = cca.fit_transform(X, Y)
    return {n: X_c[i] for i, n in enumerate(nodes)}, cca


def cca_align(
    n2v: Dict[object, Iterable[float]],
    gw: Dict[object, Iterable[float]],
    *,
    n_components: int = 32,
    path: str = os.path.join(CACHE_ROOT, "cca.pkl"),
) -> Dict[object, np.ndarray]:
    """Return latent vectors and persist CCA weights for inference.

    The matrices ``(Wn2v, Wgw)`` are pickled together at ``path`` so
    that subsequent runs can reuse the transformation.
    """

    latent, cca = aligned_cca(n2v, gw, n_components=n_components)
    import pickle

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        # store weights using protocol 4 for compatibility
        pickle.dump({"Wn2v": cca.x_weights_, "Wgw": cca.y_weights_}, f, protocol=4)
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    logging.getLogger(__name__).info("cca_sha=%s", sha)
    return latent


def load_cca(
    path: str = os.path.join(CACHE_ROOT, "cca.pkl")
) -> tuple[np.ndarray, np.ndarray]:
    """Return CCA weights ``(Wn2v, Wgw)`` previously persisted by :func:`cca_align`.

    Parameters
    ----------
    path:
        Location of the pickled weights.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(Wn2v, Wgw)`` matrices.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """

    import pickle

    import numpy as np

    if not os.path.exists(path):
        raise FileNotFoundError(f"CCA weights not found at {path}")
    with open(path, "rb") as f:
        blob = f.read()
    h = hashlib.sha256(blob).hexdigest()
    data = pickle.loads(blob)
    W1 = np.asarray(data["Wn2v"])
    W2 = np.asarray(data["Wgw"])
    W1.setflags(write=False)
    W2.setflags(write=False)
    logging.getLogger(__name__).info("cca_sha=%s", h)
    return W1, W2


def hybrid_score(
    n2v_u: Iterable[float],
    n2v_q: Iterable[float],
    gw_u: Iterable[float],
    gw_q: Iterable[float],
    hyp_u: Iterable[float],
    hyp_q: Iterable[float],
    *,
    gamma: float = 0.5,
    eta: float = 0.25,
) -> float:
    r"""Return the hybrid similarity score between two nodes.

    The score mixes Node2Vec cosine similarity, Poincar\xe9 distance and
    GraphWave cosine similarity following the formula

    ``S = \gamma cos(n2v) + \eta (1-d_B) + (1-\gamma-\eta)(1-cos(gw))``.
    ``n2v_u`` and ``n2v_q`` should be normalized if possible.
    """
    n2v_u = np.asarray(list(n2v_u), dtype=float)
    n2v_q = np.asarray(list(n2v_q), dtype=float)
    gw_u = np.asarray(list(gw_u), dtype=float)
    gw_q = np.asarray(list(gw_q), dtype=float)
    hyp_u = np.asarray(list(hyp_u), dtype=float)
    hyp_q = np.asarray(list(hyp_q), dtype=float)

    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
        return float(np.dot(a, b) / denom)

    def _poincare_dist(x: np.ndarray, y: np.ndarray) -> float:
        nx = np.linalg.norm(x)
        ny = np.linalg.norm(y)
        diff = np.linalg.norm(x - y)
        num = 2 * diff * diff
        denom = (1.0 - nx * nx) * (1.0 - ny * ny) + 1e-9
        arg = 1.0 + num / denom
        return float(np.arccosh(max(1.0, arg)))

    cos_n2v = _cos(n2v_u, n2v_q)
    cos_gw = _cos(gw_u, gw_q)
    dist_h = _poincare_dist(hyp_u, hyp_q)
    score = (
        gamma * cos_n2v + eta * (1.0 - dist_h) + (1.0 - gamma - eta) * (1.0 - cos_gw)
    )
    return float(score)


def multiview_contrastive_loss(
    n2v: Dict[object, Iterable[float]],
    gw: Dict[object, Iterable[float]],
    hyp: Dict[object, Iterable[float]],
    *,
    tau: float = 0.1,
) -> float:
    """Return the InfoNCE loss across the three views.

    The loss contrasts Node2Vec, GraphWave and Poincar\xe9 embeddings
    of each node against negative samples from the other nodes.
    """

    nodes = n2v.keys() & gw.keys() & hyp.keys()
    if not nodes:
        return 0.0

    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
        return float(np.dot(a, b) / denom)

    loss = 0.0
    for i in nodes:
        z_i = np.asarray(n2v[i], dtype=float)
        p_j = np.asarray(gw[i], dtype=float)
        numer = np.exp(_cos(z_i, p_j) / tau)
        denom = numer
        for k in nodes:
            if k == i:
                continue
            neg = np.asarray(gw[k], dtype=float)
            denom += np.exp(_cos(z_i, neg) / tau)
        loss += -np.log(numer / denom)

        # second pair (gw vs hyp)
        z_i = np.asarray(gw[i], dtype=float)
        p_j = np.asarray(hyp[i], dtype=float)
        numer = np.exp(_cos(z_i, p_j) / tau)
        denom = numer
        for k in nodes:
            if k == i:
                continue
            neg = np.asarray(hyp[k], dtype=float)
            denom += np.exp(_cos(z_i, neg) / tau)
        loss += -np.log(numer / denom)

        # third pair (hyp vs n2v)
        z_i = np.asarray(hyp[i], dtype=float)
        p_j = np.asarray(n2v[i], dtype=float)
        numer = np.exp(_cos(z_i, p_j) / tau)
        denom = numer
        for k in nodes:
            if k == i:
                continue
            neg = np.asarray(n2v[k], dtype=float)
            denom += np.exp(_cos(z_i, neg) / tau)
        loss += -np.log(numer / denom)

    return float(loss / (3 * len(nodes)))


def meta_autoencoder(
    n2v: Dict[object, Iterable[float]],
    gw: Dict[object, Iterable[float]],
    hyp: Dict[object, Iterable[float]],
    *,
    bottleneck: int = 64,
) -> Tuple[Dict[object, np.ndarray], Dict[object, np.ndarray]]:
    """Return meta-embeddings and reconstructions with a simple linear autoencoder."""

    from sklearn.decomposition import PCA

    nodes = n2v.keys() & gw.keys() & hyp.keys()
    if not nodes:
        return {}, {}

    X = [
        np.concatenate(
            [
                np.asarray(n2v[n], dtype=float),
                np.asarray(gw[n], dtype=float),
                np.asarray(hyp[n], dtype=float),
            ]
        )
        for n in nodes
    ]
    X = np.vstack(X)
    pca = PCA(n_components=bottleneck)
    Z = pca.fit_transform(X)
    recon = pca.inverse_transform(Z)

    latent: Dict[object, np.ndarray] = {}
    reconstructed: Dict[object, np.ndarray] = {}
    for idx, node in enumerate(nodes):
        latent[node] = Z[idx]
        reconstructed[node] = recon[idx]

    return latent, reconstructed
