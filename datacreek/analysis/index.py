"""ANN indexing utilities with HNSW fallback and recall tracking."""

from __future__ import annotations

import time
from typing import Iterable, Sequence, Tuple

try:
    import faiss  # type: ignore
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore
    faiss = None  # type: ignore

from contextlib import nullcontext

from .monitoring import update_metric

try:  # optional Prometheus metrics
    from prometheus_client import Gauge, Histogram

    recall_gauge = Gauge("recall10", "ANN recall@10")
    ann_latency = Histogram(
        "ann_latency_seconds",
        "Latency of ANN queries",
        buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10),
    )
except Exception:  # pragma: no cover - optional
    recall_gauge = None  # type: ignore
    ann_latency = None
from .multiview import hybrid_score


def search_with_fallback(
    xb: "np.ndarray",
    xq: "np.ndarray",
    k: int = 10,
    *,
    index: object | None = None,
    latency_threshold: float = 0.1,
) -> Tuple[Sequence[int], float, object]:
    """Return nearest neighbors using FAISS with optional HNSW fallback."""
    if faiss is None or np is None:
        raise RuntimeError("faiss not installed")
    if index is None:
        index = faiss.IndexFlatIP(xb.shape[1])
        index.add(xb)
    start = time.monotonic()
    ctx = ann_latency.time() if ann_latency is not None else nullcontext()
    with ctx:
        _, idx = index.search(xq, k)
    latency = time.monotonic() - start
    if latency > latency_threshold and not isinstance(index, faiss.IndexHNSWFlat):
        hnsw = faiss.IndexHNSWFlat(xb.shape[1], 32)
        hnsw.hnsw.efSearch = 200
        hnsw.add(xb)
        index = hnsw
        start = time.monotonic()
        ctx = ann_latency.time() if ann_latency is not None else nullcontext()
        with ctx:
            _, idx = index.search(xq, k)
        latency = time.monotonic() - start
    return idx[0], latency, index


def recall10(
    graph,
    queries: Iterable[object],
    ground_truth: dict,
    *,
    gamma: float = 0.5,
    eta: float = 0.25,
) -> float:
    """Compute recall@10 as ``hits / 10`` and push to Prometheus."""

    hits = 0
    total = 0
    for q in queries:
        rel = set(ground_truth.get(q, []))
        if not rel:
            continue

        node_data = graph.nodes[q]
        n2v_q = node_data.get("embedding")
        gw_q = node_data.get("graphwave_embedding")
        hyp_q = node_data.get("poincare_embedding")
        if n2v_q is None or gw_q is None or hyp_q is None:
            continue

        scores = []
        for u, data in graph.nodes(data=True):
            if u == q:
                continue
            n2v_u = data.get("embedding")
            gw_u = data.get("graphwave_embedding")
            hyp_u = data.get("poincare_embedding")
            if n2v_u is None or gw_u is None or hyp_u is None:
                continue
            s = hybrid_score(
                n2v_u, n2v_q, gw_u, gw_q, hyp_u, hyp_q, gamma=gamma, eta=eta
            )
            scores.append((u, s))

        scores.sort(key=lambda x: x[1], reverse=True)
        retrieved = [u for u, _ in scores[:10]]
        hits += len(rel.intersection(retrieved))
        total += 10

    recall = hits / total if total else 0.0
    if hasattr(graph, "graph"):
        graph.graph["recall10"] = recall
    else:
        graph["recall10"] = recall
    update_metric("recall10", float(recall))
    if recall_gauge is not None:
        try:
            recall_gauge.set(float(recall))
        except Exception:  # pragma: no cover
            pass
    return recall
