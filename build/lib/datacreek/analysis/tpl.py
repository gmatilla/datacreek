# Heavy topology computations. Tests operate on tiny graphs to keep them fast.
"""Topological Perception Layer utilities."""

from __future__ import annotations

from typing import Dict, Iterable

import networkx as nx
import numpy as np

try:  # optional dependency
    import gudhi as gd
except Exception:  # pragma: no cover - optional
    gd = None  # type: ignore

from .fractal import persistence_wasserstein_distance
from .generation import generate_graph_rnn_like
from .sheaf import resolve_sheaf_obstruction


def sinkhorn_w1(
    d1: np.ndarray, d2: np.ndarray, eps: float = 0.1, n_iter: int = 50
) -> float:
    """Return Sinkhorn approximation of Wasserstein-1 between diagrams.

    Uses a Sinkhorn iteration with an infinity-norm cost matrix to
    approximate the Wasserstein-1 distance.
    """

    if d1.size == 0 or d2.size == 0:
        return 0.0

    a = np.ones(len(d1)) / len(d1)
    b = np.ones(len(d2)) / len(d2)
    # cost matrix uses L_inf on birth/death pairs
    C = np.maximum(
        np.abs(d1[:, None, 0] - d2[None, :, 0]),
        np.abs(d1[:, None, 1] - d2[None, :, 1]),
    )
    K = np.exp(-C / eps)
    u = np.ones_like(a)
    for _ in range(n_iter):
        v = b / (K.T @ u)
        u = a / (K @ v)
    transport = np.outer(u, v) * K
    return float(np.sum(transport * C))


def _diagram(graph: nx.Graph, dimension: int = 1) -> np.ndarray:
    """Return persistence diagram of ``graph`` in ``dimension``."""

    if gd is None:
        raise RuntimeError("gudhi is required for persistence calculations")

    st = gd.SimplexTree()
    mapping = {n: i for i, n in enumerate(graph.nodes())}
    for node, idx in mapping.items():
        st.insert([idx], filtration=0.0)
    for u, v in graph.edges():
        st.insert([mapping[u], mapping[v]], filtration=1.0)

    st.compute_persistence(persistence_dim_max=True)
    diag = np.array(st.persistence_intervals_in_dimension(dimension))
    if diag.size == 0:
        return np.empty((0, 2))
    diag = diag[np.isfinite(diag[:, 1])]
    return diag


def tpl_correct_graph(
    graph: nx.Graph,
    target: nx.Graph,
    *,
    epsilon: float = 0.1,
    dimension: int = 1,
    order: int = 1,
    max_iter: int = 5,
) -> Dict[str, float | bool]:
    """Correct ``graph`` topology if Wasserstein-1 distance is too high."""

    d1 = _diagram(graph, dimension)
    d2 = _diagram(target, dimension)
    dist_before = sinkhorn_w1(d1, d2)

    corrected = False
    if dist_before > epsilon:
        motif = generate_graph_rnn_like(
            target.number_of_nodes(), target.number_of_edges()
        )
        mapping = {i: n for i, n in enumerate(graph.nodes())}
        for u, v in motif.edges():
            a = mapping.get(u)
            b = mapping.get(v)
            if a is None or b is None:
                continue
            if not graph.has_edge(a, b):
                graph.add_edge(a, b, relation="perception_link")
        resolve_sheaf_obstruction(graph, max_iter=max_iter)
        corrected = True

    d1_after = _diagram(graph, dimension)
    dist_after = sinkhorn_w1(d1_after, d2)

    return {
        "distance_before": float(dist_before),
        "distance_after": float(dist_after),
        "corrected": corrected,
    }
