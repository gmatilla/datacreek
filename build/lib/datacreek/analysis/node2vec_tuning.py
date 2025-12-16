# Heavy optimization utilities for node2vec tuning. Requires large
# graph datasets and GPU/FAISS support which may slow down tests.
from __future__ import annotations

import hashlib
import json
import pathlib
import time
from typing import Dict, Iterable, Sequence

import numpy as np
from skopt import Optimizer

from .index import recall10

# JSON artifact tracking the best Node2Vec parameters for the last run.
BEST_PQ_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "benchmarks" / "best_pq.json"
)


def _var_norm(graph: "KnowledgeGraph") -> float:
    """Return variance of embedding norms stored on ``graph``."""
    embs = [
        np.asarray(data["embedding"], dtype=float)
        for _, data in graph.graph.nodes(data=True)
        if "embedding" in data
    ]
    if not embs:
        return float("inf")
    norms = np.linalg.norm(np.vstack(embs), axis=1)
    return float(np.var(norms))


def _dataset_hash(graph: "KnowledgeGraph") -> str:
    """Return an MD5 hash summarizing the dataset."""

    m = hashlib.md5()
    m.update("".join(sorted(str(n) for n in graph.graph.nodes())).encode())
    m.update("".join(sorted(f"{u}-{v}" for u, v in graph.graph.edges())).encode())
    return m.hexdigest()


def _save_artifact(path: pathlib.Path, dataset: str, p: float, q: float) -> None:
    """Persist best parameters to JSON ``path``."""

    try:
        path.write_text(json.dumps({"dataset": dataset, "p": p, "q": q}))
    except Exception:  # pragma: no cover - best effort
        pass


def autotune_node2vec(
    kg: "KnowledgeGraph",
    queries: Sequence[object],
    ground_truth: Dict[object, Sequence[object]],
    *,
    k: int = 10,
    var_threshold: float = 1e-4,
    max_evals: int = 40,
    max_minutes: float = 30.0,
) -> tuple[float, float]:
    """Tune Node2Vec ``p`` and ``q`` using Bayesian optimisation."""

    opt = Optimizer([(0.1, 4.0), (0.1, 4.0)], base_estimator="GP", acq_func="EI")
    best_recall = -1.0
    best_pq = (1.0, 1.0)
    since_improve = 0
    start_t = time.monotonic()

    for _ in range(max_evals):
        cand = opt.ask()
        p, q = float(cand[0]), float(cand[1])
        kg.compute_node2vec_embeddings(p=p, q=q)
        rec = recall10(kg.graph, queries, ground_truth, gamma=0.5, eta=0.25)
        var_n = _var_norm(kg)
        opt.tell(cand, -rec)
        if rec > best_recall:
            best_recall = rec
            best_pq = (p, q)
            since_improve = 0
        else:
            since_improve += 1
        if (
            var_n < var_threshold
            or since_improve >= 5
            or (time.monotonic() - start_t) / 60.0 > max_minutes
        ):
            break

    kg.compute_node2vec_embeddings(p=best_pq[0], q=best_pq[1])
    recall10(kg.graph, queries, ground_truth, gamma=0.5, eta=0.25)
    _save_artifact(BEST_PQ_PATH, _dataset_hash(kg), best_pq[0], best_pq[1])
    return best_pq
