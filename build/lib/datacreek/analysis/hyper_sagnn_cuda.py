# pragma: no cover - requires GPU libraries
"""CUDA-accelerated Hyper-SAGNN utilities."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from .hypergraph import hyper_sagnn_embeddings


def hyper_sagnn_embeddings_stream(
    hyperedges: Iterable[Sequence[int]],
    node_features: np.ndarray,
    *,
    embed_dim: int | None = None,
    stream_batch: int = 8,
    device: str = "cuda",
    seed: int | None = None,
) -> np.ndarray:
    """Return Hyper-SAGNN embeddings using CUDA streams when available.

    Parameters
    ----------
    hyperedges:
        Collection of hyperedges given as sequences of node indices.
    node_features:
        Feature matrix of shape ``(num_nodes, feat_dim)``.
    embed_dim:
        Dimension of the output embedding. Defaults to the feature dimension.
    stream_batch:
        Number of hyperedges processed per CUDA stream.
    device:
        Compute device. ``"cuda"`` uses GPU streams when ``torch`` and CUDA are
        available. ``"cpu"`` falls back to a NumPy implementation.
    seed:
        Optional random seed controlling the attention weights.
    """

    try:
        import torch
    except Exception:  # pragma: no cover - torch missing
        torch = None  # type: ignore

    use_cuda = (
        device == "cuda"
        and torch is not None
        and getattr(torch.cuda, "is_available", lambda: False)()
    )
    if not use_cuda:
        # fallback to pure NumPy implementation
        return hyper_sagnn_embeddings(
            hyperedges, node_features, embed_dim=embed_dim, seed=seed
        )

    feat_dim = node_features.shape[1]
    d = embed_dim or feat_dim

    features = torch.as_tensor(node_features, dtype=torch.float32, device=device)

    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
    W_q = torch.randn(feat_dim, d, generator=rng, device=device) / math.sqrt(feat_dim)
    W_k = torch.randn(feat_dim, d, generator=rng, device=device) / math.sqrt(feat_dim)
    W_v = torch.randn(feat_dim, d, generator=rng, device=device) / math.sqrt(feat_dim)

    edges = list(hyperedges)
    results = [torch.empty(d, device="cpu") for _ in edges]
    num_streams = max(1, (len(edges) + stream_batch - 1) // stream_batch)
    streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]

    for idx, edge in enumerate(edges):
        stream = streams[idx // stream_batch]
        idxs = torch.tensor(edge, device=device)
        with torch.cuda.stream(stream):
            X = features.index_select(0, idxs)
            q = X @ W_q
            k = X @ W_k
            v = X @ W_v
            scores = (q @ k.T) / math.sqrt(d)
            weights = torch.softmax(scores, dim=1)
            context = weights @ v
            emb = context.mean(dim=0)
            results[idx] = emb.to("cpu")

    for stream in streams:
        stream.synchronize()

    return torch.stack(results).numpy()
