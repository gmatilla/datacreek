# Incremental topology utilities typically require extensive data but
# remain testable on small graphs.
from __future__ import annotations

from typing import Dict

import networkx as nx
import numpy as np

try:  # optional dependency
    import gudhi as gd
except Exception:  # pragma: no cover - optional
    gd = None  # type: ignore

__all__ = ["tpl_incremental"]


def _local_hash(graph: nx.Graph, node: int, radius: int) -> int:
    """Return hash of the ``radius``-hop neighbourhood of ``node``.

    The hash includes edges and a ``timestamp`` attribute when present so that
    updates to connectivity are detected without recomputing all nodes.
    """

    neigh = nx.single_source_shortest_path_length(graph, node, cutoff=radius)
    nodes = set(neigh)
    edges: list[tuple[int, int, float]] = []
    for u, v, data in graph.edges(nodes, data=True):
        if u in nodes and v in nodes:
            a, b = sorted((u, v))
            ts = float(data.get("timestamp", 0.0))
            edges.append((a, b, ts))

    return hash((tuple(sorted(nodes)), tuple(sorted(edges))))


def _local_persistence(
    graph: nx.Graph, node: int, *, radius: int = 1, dimension: int = 1
) -> np.ndarray:
    """Return persistence diagram of ``node`` neighbourhood."""
    if gd is None:
        raise RuntimeError("gudhi is required for persistence calculations")
    nodes = nx.single_source_shortest_path_length(graph, node, cutoff=radius).keys()
    sub = graph.subgraph(nodes)
    st = gd.SimplexTree()
    mapping = {n: i for i, n in enumerate(sub.nodes())}
    for n, idx in mapping.items():
        st.insert([idx], filtration=0.0)
    for u, v in sub.edges():
        st.insert([mapping[u], mapping[v]], filtration=1.0)
    st.compute_persistence(persistence_dim_max=True)
    diag = np.array(st.persistence_intervals_in_dimension(dimension))
    if diag.size == 0:
        return np.empty((0, 2))
    diag = diag[np.isfinite(diag[:, 1])]
    return diag


def tpl_incremental(
    graph: nx.Graph, *, radius: int = 1, dimension: int = 1
) -> Dict[int, np.ndarray]:
    """Update and return local persistence diagrams for ``graph`` nodes."""

    diags: Dict[int, np.ndarray] = {}
    for node in graph.nodes():
        h = _local_hash(graph, node, radius)
        if graph.nodes[node].get("tpl_hash") != h:
            diag = _local_persistence(graph, node, radius=radius, dimension=dimension)
            graph.nodes[node]["tpl_diag"] = diag.tolist()
            graph.nodes[node]["tpl_hash"] = h
        data = graph.nodes[node].get("tpl_diag")
        if data is not None:
            diags[node] = np.asarray(data, dtype=float)

    if diags:
        global_diag = (
            np.concatenate(list(diags.values())) if diags else np.empty((0, 2))
        )
        graph.graph["tpl_global"] = global_diag.tolist()

    return diags
