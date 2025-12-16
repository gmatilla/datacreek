#!/usr/bin/env python
"""Benchmark ANN backends.

This script builds an index from random vectors and measures P95 search
latency and recall for k-NN queries. Example usage::

    python scripts/ann_benchmark.py --backend faiss_gpu_ivfpq
"""
from __future__ import annotations

import argparse
import random
import time
from statistics import quantiles

import numpy as np

from datacreek.core.knowledge_graph import KnowledgeGraph


def benchmark(backend: str, n: int = 1000, d: int = 128, k: int = 5) -> None:
    """Build a graph with random embeddings and benchmark the ANN backend."""
    kg = KnowledgeGraph()
    for i in range(n):
        kg.graph.add_node(f"n{i}", embedding=np.random.rand(d).tolist())

    kg.build_faiss_index(method=backend)

    q = np.random.rand(d)
    latencies = []
    recall_hits = 0
    true_ids = list(range(n))
    vectors = np.vstack([kg.graph.nodes[n]["embedding"] for n in kg.graph.nodes])
    base = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q))
    true_top = set(np.argsort(base)[-k:])

    for _ in range(100):
        start = time.monotonic()
        res = kg.search_faiss(q, k=k)
        latencies.append(time.monotonic() - start)
        idxs = [int(r[1:]) for r in res]
        recall_hits += len(true_top.intersection(idxs))

    p95 = quantiles(latencies, n=100)[94]
    recall = recall_hits / (100 * k)
    print(f"backend={backend} p95={p95:.4f}s recall={recall:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="flat")
    args = parser.parse_args()
    benchmark(args.backend)


if __name__ == "__main__":
    main()
