# Heavy FAISS tuning logic. Runs slowly on large datasets but can
# be exercised on tiny inputs during testing.
"""Bayesian optimization of FAISS IVFPQ ``nprobe`` for recall@100."""

from __future__ import annotations

import pathlib
import pickle
import time
from typing import Sequence

import numpy as np
from skopt import Optimizer

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - faiss optional
    faiss = None  # type: ignore


def _compute_recall(
    result_ids: np.ndarray, ground_truth: Sequence[Sequence[int]]
) -> float:
    """Return mean recall over all queries."""
    total = 0.0
    hits = 0
    for idx, gt in zip(result_ids, ground_truth):
        gt_set = set(gt)
        hits += len(gt_set.intersection(idx))
        total += len(gt_set)
    return hits / total if total else 0.0


def autotune_nprobe(
    index: "faiss.IndexIVFPQ",
    xb: np.ndarray,
    xq: np.ndarray,
    *,
    k: int = 100,
    target: float = 0.92,
    max_evals: int = 40,
) -> int:
    """Tune ``nprobe`` using Bayesian optimisation over recall@100.

    Parameters
    ----------
    index:
        IVFPQ index to tune.
    xb:
        Database vectors used to build ground-truth nearest neighbours.
    xq:
        Query vectors on which to evaluate recall.
    k:
        Number of neighbours used when computing recall.
    target:
        Target recall value triggering early stop.
    max_evals:
        Maximum number of evaluations of the objective function.

    Returns
    -------
    int
        Best ``nprobe`` value found.
    """
    if faiss is None:
        raise RuntimeError("faiss not installed")

    flat = faiss.IndexFlatIP(xb.shape[1])
    flat.add(xb)
    _, gt_idx = flat.search(xq, k)

    opt = Optimizer([(32, 256)], base_estimator="GP", acq_func="EI")
    best_recall = 0.0
    best_nprobe = index.nprobe

    for _ in range(max_evals):
        [cand] = opt.ask()
        nprobe = int(cand)
        index.nprobe = nprobe
        _, idx = index.search(xq, k)
        recall = _compute_recall(idx, gt_idx)
        opt.tell([cand], -recall)
        if recall > best_recall:
            best_recall = recall
            best_nprobe = nprobe
        if recall >= target:
            break

    index.nprobe = best_nprobe
    return best_nprobe


def profile_nprobe(
    index: "faiss.IndexIVFPQ",
    xb: np.ndarray,
    xq: np.ndarray,
    *,
    k: int = 100,
    nprobes: Sequence[int] | None = None,
    path: str | pathlib.Path | None = None,
) -> dict[str, list[float]]:
    """Return recall and latency curves for a range of ``nprobe`` values.

    Parameters
    ----------
    index:
        IVFPQ index to evaluate.
    xb:
        Database vectors to compute ground truth.
    xq:
        Query vectors.
    k:
        Recall computed over ``k`` neighbours.
    nprobes:
        Iterable of ``nprobe`` values to test (defaults to ``32..256`` step 32).
    path:
        Optional path where to ``pickle`` the resulting dictionary.

    Returns
    -------
    dict[str, list[float]]
        Mapping with keys ``nprobe``, ``latency`` and ``recall``.
    """
    if faiss is None:
        raise RuntimeError("faiss not installed")

    if nprobes is None:
        nprobes = range(32, 257, 32)

    flat = faiss.IndexFlatIP(xb.shape[1])
    flat.add(xb)
    _, gt_idx = flat.search(xq, k)

    out = {"nprobe": [], "latency": [], "recall": []}
    for nprobe in nprobes:
        index.nprobe = int(nprobe)
        start = time.monotonic()
        _, idx = index.search(xq, k)
        elapsed = time.monotonic() - start
        out["nprobe"].append(int(nprobe))
        out["latency"].append(elapsed)
        out["recall"].append(_compute_recall(idx, gt_idx))

    if path is not None:
        with open(path, "wb") as fh:
            pickle.dump(out, fh)

    return out
