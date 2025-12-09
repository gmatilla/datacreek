import math
from typing import Iterable, List, Sequence

import numpy as np


def hyper_sagnn_embeddings(
    hyperedges: List[Sequence[int]],
    node_features: np.ndarray,
    *,
    embed_dim: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Return Hyper-SAGNN style embeddings for each hyperedge.

    Parameters
    ----------
    hyperedges:
        List of hyperedges, each given as a sequence of node indices.
    node_features:
        Array of shape ``(num_nodes, feat_dim)`` containing node features.
    embed_dim:
        Dimension ``d`` of the output embedding. Defaults to ``feat_dim``.
    seed:
        Optional random seed controlling the internal attention weights.
    """
    rng = np.random.default_rng(seed)
    feat_dim = node_features.shape[1]
    d = embed_dim or feat_dim

    W_q = rng.normal(scale=1.0 / math.sqrt(feat_dim), size=(feat_dim, d))
    W_k = rng.normal(scale=1.0 / math.sqrt(feat_dim), size=(feat_dim, d))
    W_v = rng.normal(scale=1.0 / math.sqrt(feat_dim), size=(feat_dim, d))

    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_max = x.max(axis=axis, keepdims=True)
        e = np.exp(x - x_max)
        return e / e.sum(axis=axis, keepdims=True)

    embeddings = []
    for edge in hyperedges:
        X = node_features[np.asarray(edge)]
        q = X @ W_q
        k = X @ W_k
        v = X @ W_v
        scores = q @ k.T / math.sqrt(d)
        weights = _softmax(scores, axis=1)
        context = weights @ v
        emb = context.mean(axis=0)
        embeddings.append(emb)
    return np.stack(embeddings)


def hyper_sagnn_head_drop_embeddings(
    hyperedges: List[Sequence[int]],
    node_features: np.ndarray,
    *,
    num_heads: int = 4,
    threshold: float = 0.1,
    seed: int | None = None,
) -> np.ndarray:
    """Return Hyper-SAGNN embeddings with head-drop pruning.

    Parameters
    ----------
    hyperedges:
        List of hyperedges as sequences of node indices.
    node_features:
        Matrix of shape ``(num_nodes, feat_dim)``.
    num_heads:
        Number of attention heads.
    threshold:
        Minimum absolute mean attention weight to keep a head.
    seed:
        Optional random seed controlling the attention parameters.

    Returns
    -------
    np.ndarray
        Array of shape ``(len(hyperedges), head_dim)`` with pruned embeddings.
    """
    rng = np.random.default_rng(seed)
    feat_dim = node_features.shape[1]
    head_dim = max(1, feat_dim // num_heads)

    W_q = rng.normal(
        scale=1.0 / math.sqrt(feat_dim), size=(num_heads, feat_dim, head_dim)
    )
    W_k = rng.normal(
        scale=1.0 / math.sqrt(feat_dim), size=(num_heads, feat_dim, head_dim)
    )
    W_v = rng.normal(
        scale=1.0 / math.sqrt(feat_dim), size=(num_heads, feat_dim, head_dim)
    )

    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_max = x.max(axis=axis, keepdims=True)
        e = np.exp(x - x_max)
        return e / e.sum(axis=axis, keepdims=True)

    embeddings = []
    for edge in hyperedges:
        X = node_features[np.asarray(edge)]
        head_embs = []
        for h in range(num_heads):
            q = X @ W_q[h]
            k = X @ W_k[h]
            v = X @ W_v[h]
            scores = q @ k.T / math.sqrt(head_dim)
            weights = _softmax(scores, axis=1)
            importance = float(np.abs(weights).mean())
            if importance < threshold:
                continue
            context = weights @ v
            head_embs.append(context.mean(axis=0))
        if head_embs:
            emb = np.mean(head_embs, axis=0)
        else:
            emb = np.zeros(head_dim)
        embeddings.append(emb)
    return np.stack(embeddings)


def hyper_adamic_adar_scores(
    hyperedges: Iterable[Iterable[object]],
) -> dict[tuple[object, object], float]:
    r"""Return Hyper-Adamic\N{EN DASH}Adar scores between pairs of nodes.

    Each hyperedge contributes :math:`1/\log(|H|-1)` to every pair of its
    incident nodes. This generalizes the classical Adamic--Adar index
    to higher-order interactions by penalising larger hyperedges via the
    logarithm of their size minus one.

    Parameters
    ----------
    hyperedges:
        Collection of hyperedges, each given as an iterable of node IDs.

    Returns
    -------
    dict
        Mapping ``(u, v)`` to accumulated score.
    """

    from itertools import combinations

    scores: dict[tuple[object, object], float] = {}
    for edge in hyperedges:
        nodes = list(dict.fromkeys(edge))
        if len(nodes) < 2:
            continue
        # Weight following Hyper-AA: 1 / log(|H|-1)
        denom = max(2, len(nodes)) - 1
        if denom <= 1:
            weight = 0.0
        else:
            weight = 1.0 / math.log(denom)
        for u, v in combinations(nodes, 2):
            pair = (u, v) if u <= v else (v, u)
            scores[pair] = scores.get(pair, 0.0) + weight
    return scores


def hyperedge_attention_scores(
    hyperedges: List[Sequence[int]],
    node_features: np.ndarray,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Return an attention-based importance score for each hyperedge.

    For pruning and prioritization we compute the average absolute
    attention weight produced by a single-head self-attention layer.
    Higher scores indicate that the hyperedge attracts more attention.

    Parameters
    ----------
    hyperedges:
        List of hyperedges, each as a sequence of node indices.
    node_features:
        Feature matrix of shape ``(num_nodes, feat_dim)``.
    seed:
        Optional seed controlling the random attention parameters.

    Returns
    -------
    np.ndarray
        Array of shape ``(len(hyperedges),)`` with attention scores.
    """
    rng = np.random.default_rng(seed)
    feat_dim = node_features.shape[1]

    W_q = rng.normal(scale=1.0 / math.sqrt(feat_dim), size=(feat_dim, 1))
    W_k = rng.normal(scale=1.0 / math.sqrt(feat_dim), size=(feat_dim, 1))

    scores = []
    for edge in hyperedges:
        X = node_features[np.asarray(edge)]
        q = X @ W_q  # (|H|, 1)
        k = X @ W_k  # (|H|, 1)
        # attention weights for single head
        att = q @ k.T
        att = np.exp(att - att.max(axis=1, keepdims=True))
        att = att / att.sum(axis=1, keepdims=True)
        scores.append(float(np.abs(att).mean()))
    return np.asarray(scores)
