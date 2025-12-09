"""Benchmark Hybrid ANN recall and latency."""

import argparse
import importlib.abc
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

try:  # optional dependency
    import faiss
except Exception:  # pragma: no cover - faiss may be missing on CI
    faiss = None  # type: ignore


SPEC = importlib.util.spec_from_file_location(
    "hybrid_ann",
    Path(__file__).resolve().parents[1] / "datacreek" / "analysis" / "hybrid_ann.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert isinstance(SPEC.loader, importlib.abc.Loader)
SPEC.loader.exec_module(MODULE)  # type: ignore
search_hnsw_pq = MODULE.search_hnsw_pq


def run_bench(
    n: int = 1_000_000,
    d: int = 256,
    k: int = 100,
    queries: int = 1000,
    *,
    threads: int = 32,
) -> dict:
    """Benchmark Hybrid ANN recall and latency.

    Parameters
    ----------
    n:
        Number of database vectors. Defaults to ``1_000_000`` as required by the
        benchmark spec.
    d:
        Vector dimensionality. Defaults to ``256``.
    k:
        Number of neighbours queried.
    queries:
        Number of query vectors.
    threads:
        Number of CPU threads used by FAISS during search.
    """
    rng = np.random.default_rng(0)
    xb = rng.standard_normal((n, d)).astype(np.float32)
    if faiss is not None:
        faiss.omp_set_num_threads(threads)
    xq = xb[:queries] + 0.01

    gt_dist = np.linalg.norm(xb[None, :] - xq[:, None, :], axis=2)
    gt = np.argsort(gt_dist, axis=1)[:, :k]

    times = []
    recall = []
    for i in range(queries):
        t0 = time.perf_counter()
        idx = search_hnsw_pq(xb, xq[i : i + 1], k=k, prefetch=500)
        times.append(time.perf_counter() - t0)
        recall.append(len(set(idx) & set(gt[i])) / k)

    return {
        f"recall@{k}": float(np.mean(recall)),
        "p95_ms": float(np.percentile(np.array(times) * 1000, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="bench_hybrid_ann.json")
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--d", type=int, default=256)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    res = run_bench(args.n, args.d, args.k, args.queries, threads=args.threads)
    with open(args.output, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
