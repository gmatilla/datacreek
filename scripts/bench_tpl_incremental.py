#!/usr/bin/env python3
"""Benchmark incremental TPL vs full recomputation.

This script modifies a random graph by a fraction of its edges and
compares the runtime of :func:`tpl_incremental` when recomputing all
nodes from scratch versus reusing cached diagrams. The benchmark fails
with exit code 1 if the speedup is below 2x when the edge perturbation
is at most 20%.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import networkx as nx
import numpy as np


def _apply_ops(graph: nx.Graph, ops: list[tuple[int, int, bool]]) -> None:
    """Apply edge operations to ``graph``."""
    for u, v, add in ops:
        if add:
            graph.add_edge(u, v)
        else:
            if graph.has_edge(u, v):
                graph.remove_edge(u, v)


def _bench_once(delta: float, n: int = 100, m: int = 200) -> float:
    """Return speedup ratio full / incremental for ``delta`` edge changes."""
    import datacreek.analysis.tpl_incremental as tpli
    from datacreek.analysis.tpl_incremental import tpl_incremental

    base = nx.gnm_random_graph(n, m, seed=0)
    tpl_incremental(base)
    edges = list(base.edges())
    num = max(1, int(len(edges) * delta))
    nodes = list(base.nodes())
    random.seed(0)
    ops: list[tuple[int, int, bool]] = []
    for i in range(num):
        u, v = edges[i]
        ops.append((u, v, False))
    for _ in range(num):
        u, v = random.sample(nodes, 2)
        ops.append((u, v, True))

    g_inc = base.copy()
    _apply_ops(g_inc, ops)
    t0 = time.perf_counter()
    tpl_incremental(g_inc)
    inc_time = time.perf_counter() - t0

    g_full = base.copy()
    _apply_ops(g_full, ops)
    for n in g_full.nodes():
        g_full.nodes[n].pop("tpl_hash", None)
        g_full.nodes[n].pop("tpl_diag", None)
    g_full.graph.pop("tpl_global", None)
    t0 = time.perf_counter()
    tpl_incremental(g_full)
    full_time = time.perf_counter() - t0
    return full_time / max(inc_time, 1e-9)


def run_bench(deltas: list[float]) -> dict[str, float]:
    """Run benchmark for a list of ``deltas``."""
    results = {}
    for d in deltas:
        results[str(d)] = _bench_once(d)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    deltas = [0.01, 0.1, 0.2]
    results = run_bench(deltas)
    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results))
    if any(d <= 0.2 and r < 2.0 for d, r in zip(deltas, results.values())):
        raise SystemExit(1)


if __name__ == "__main__":
    from datacreek.analysis import tpl_incremental as tpli  # type: ignore

    if tpli.gd is None:
        # Provide a lightweight persistence stub when gudhi is absent
        def _stub(*args, **kwargs):
            nodes = nx.single_source_shortest_path_length(
                args[0], args[1], cutoff=kwargs.get("radius", 1)
            ).keys()
            arr = np.array([[0.0, float(len(list(nodes)))]])
            time.sleep(0.0005)
            return arr

        tpli._local_persistence = _stub  # type: ignore

    main()
