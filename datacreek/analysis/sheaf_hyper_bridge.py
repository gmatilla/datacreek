from __future__ import annotations

"""Bridge between sheaf and hypergraph spectra."""

import networkx as nx
import numpy as np

from .monitoring import update_metric
from .sheaf import sheaf_incidence_matrix, sheaf_laplacian


def sheaf_hyper_incidence(
    graph: nx.Graph, *, edge_attr: str = "sheaf_sign"
) -> np.ndarray:
    """Return signed incidence matrix from a sheaf graph."""
    return sheaf_incidence_matrix(graph, edge_attr=edge_attr)


def sheaf_hyper_bridge_score(
    graph: nx.Graph, *, edge_attr: str = "sheaf_sign"
) -> float:
    r"""Return spectral coherence score between sheaf and hypergraph views.

    The score compares eigenvalues of the sheaf Laplacian and the hypergraph
    Laplacian constructed from the signed incidence matrix of ``graph``.
    It is defined as

    .. math::

       S = 1 - \frac{\sum_i |\lambda_i^{\text{sheaf}} - \lambda_i^{\text{hyper}}|}
                {\sum_i \lambda_i^{\text{sheaf}} + \lambda_i^{\text{hyper}}}.

    Parameters
    ----------
    graph:
        Input graph with sheaf sign information.
    edge_attr:
        Name of the edge attribute storing the sheaf sign.

    Returns
    -------
    float
        Coherence score between 0 and 1 (higher is better).
    """

    B = sheaf_incidence_matrix(graph, edge_attr=edge_attr)
    if B.size == 0:
        return 0.0
    L_s = sheaf_laplacian(graph, edge_attr=edge_attr)
    L_h = B @ B.T
    eig_s = np.linalg.eigvalsh(L_s)
    eig_h = np.linalg.eigvalsh(L_h)
    k = min(len(eig_s), len(eig_h))
    diff = float(np.sum(np.abs(eig_s[:k] - eig_h[:k])))
    total = float(np.sum(eig_s[:k]) + np.sum(eig_h[:k]) + 1e-12)
    score = 1.0 - diff / total
    try:
        update_metric("sheaf_hyper_score", score)
    except Exception:
        pass  # metric or monitoring not available
    return score


def top_k_incoherent(
    graph: nx.Graph,
    k: int,
    tau: float,
    *,
    edge_attr: str = "sheaf_sign",
) -> list[tuple[tuple[int, int], float]]:
    """Return ``k`` edges whose removal increases spectral mismatch.

    For each edge we remove it and recompute eigenvalues of the sheaf and
    hypergraph Laplacians. The change in total absolute eigenvalue difference
    defines ``Δλ_e``. Edges with ``Δλ_e > τ`` are considered incoherent and
    returned sorted by this value.

    Parameters
    ----------
    graph:
        Input graph with sheaf sign information.
    k:
        Maximum number of edges to return.
    tau:
        Minimum eigenvalue difference ``Δλ_e`` to include an edge.
    edge_attr:
        Name of the edge attribute storing the sheaf sign.

    Returns
    -------
    list of ((int, int), float)
        Edge tuples with their ``Δλ_e`` value sorted descending.
    """

    B = sheaf_incidence_matrix(graph, edge_attr=edge_attr)
    if B.size == 0:
        return []
    L_s = sheaf_laplacian(graph, edge_attr=edge_attr)
    L_h = B @ B.T
    base = float(np.sum(np.abs(np.linalg.eigvalsh(L_s) - np.linalg.eigvalsh(L_h))))

    incoherent: list[tuple[tuple[int, int], float]] = []
    for u, v in graph.edges():
        g2 = graph.copy()
        g2.remove_edge(u, v)
        B2 = sheaf_incidence_matrix(g2, edge_attr=edge_attr)
        if B2.size == 0:
            continue
        L_s2 = sheaf_laplacian(g2, edge_attr=edge_attr)
        L_h2 = B2 @ B2.T
        diff = float(
            np.sum(np.abs(np.linalg.eigvalsh(L_s2) - np.linalg.eigvalsh(L_h2)))
        )
        delta = abs(diff - base)
        if delta > tau:
            incoherent.append(((u, v), delta))

    incoherent.sort(key=lambda x: x[1], reverse=True)
    return incoherent[:k]
