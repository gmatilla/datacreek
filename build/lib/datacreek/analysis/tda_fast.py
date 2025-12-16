"""Fast persistence diagram computations for graphs."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, Tuple

import networkx as nx
import numpy as np


class _UnionFind:
    """Disjoint-set data structure for union-find operations."""

    def __init__(self, nodes: Iterable[int]) -> None:
        self.parent = {n: n for n in nodes}
        self.rank = {n: 0 for n in nodes}

    def find(self, x: int) -> int:
        parent = self.parent
        if parent[x] != x:
            parent[x] = self.find(parent[x])
        return parent[x]

    def union(self, x: int, y: int) -> bool:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        rank = self.rank
        if rank[root_x] < rank[root_y]:
            root_x, root_y = root_y, root_x
        self.parent[root_y] = root_x
        if rank[root_x] == rank[root_y]:
            rank[root_x] += 1
        return True


def _compute_fast_diagrams(graph: nx.Graph, max_dim: int) -> Dict[int, np.ndarray]:
    """Return persistence diagrams up to ``max_dim`` using a fast union-find approach."""

    edges = [(data.get("weight", 1.0), u, v) for u, v, data in graph.edges(data=True)]
    edges.sort()

    uf = _UnionFind(graph.nodes())
    diag0 = []
    cycle_edges = []
    for w, u, v in edges:
        if uf.union(u, v):
            diag0.append((0.0, float(w)))
        else:
            if max_dim >= 1:
                cycle_edges.append((float(w)))

    max_w = max((float(w) for w, *_ in edges), default=1.0)
    diag1 = [(w, max_w) for w in cycle_edges] if max_dim >= 1 else []

    return {
        0: np.asarray(diag0, dtype=float),
        1: np.asarray(diag1, dtype=float),
    }


@lru_cache(maxsize=256)
def _fast_diagrams_cached(
    nodes: Tuple[int, ...], edges: Tuple[Tuple[int, int, float], ...], max_dim: int
) -> Dict[int, np.ndarray]:
    """LRU-cached wrapper building a graph from ``nodes`` and ``edges``."""

    g = nx.Graph()
    g.add_nodes_from(nodes)
    for u, v, w in edges:
        g.add_edge(u, v, weight=w)
    return _compute_fast_diagrams(g, max_dim)


def fast_persistence_diagrams(
    graph: nx.Graph, max_dim: int = 1
) -> Dict[int, np.ndarray]:
    """Return fast persistence diagrams for ``graph``."""

    nodes = tuple(sorted(graph.nodes()))
    edges = tuple(
        sorted(
            (
                min(u, v),
                max(u, v),
                float(data.get("weight", 1.0)),
            )
            for u, v, data in graph.edges(data=True)
        )
    )
    return _fast_diagrams_cached(nodes, edges, max_dim)
