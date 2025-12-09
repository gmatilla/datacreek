"""Utilities to propose graph edge repair suggestions.

This module scans the knowledge graph for edges whose removal would
significantly decrease the spectral discrepancy between the sheaf and
hypergraph structures.  Edges with a change in eigenvalue ``Δλ`` above a
threshold ``tau`` and a sheaf consistency score ``S`` greater than
``s_threshold`` generate ``MERGE``/``DELETE`` Cypher patches that are
stored in the ``repair_suggestions`` collection.

The job is intended to run periodically via cron and write suggestions to
Redis so operators can review and apply them through the ``/explain``
API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

import networkx as nx

from datacreek.analysis import sheaf_consistency_score, top_k_incoherent
from datacreek.backends import get_neo4j_driver, get_redis_client
from datacreek.core.dataset import DatasetBuilder


@dataclass
class RepairSuggestion:
    """Container for a suggested patch on an incoherent edge."""

    u: str
    v: str
    delta: float
    delete: str
    merge: str

    def to_json(self) -> str:
        """Return the suggestion serialised as JSON."""
        return json.dumps(
            {
                "u": self.u,
                "v": self.v,
                "delta": self.delta,
                "delete": self.delete,
                "merge": self.merge,
            }
        )


def collect_incoherent_edges(
    g: nx.Graph,
    top: int = 5,
    tau: float = 0.0,
    s_threshold: float = 0.8,
) -> List[RepairSuggestion]:
    """Return repair suggestions for edges exceeding ``tau`` and ``s_threshold``.

    Parameters
    ----------
    g:
        Graph to inspect. The graph is temporarily modified when evaluating
        sheaf consistency for each candidate edge.
    top:
        Maximum number of edges returned by :func:`top_k_incoherent`.
    tau:
        Minimum eigenvalue delta ``Δλ`` for an edge to be considered.
    s_threshold:
        Lower bound on the sheaf consistency score ``S`` after removing the
        edge. Only edges that *improve* the score beyond this threshold are
        retained.
    """

    suggestions: List[RepairSuggestion] = []
    for (u, v), delta in top_k_incoherent(g, top, tau):
        if not g.has_edge(u, v):
            continue
        g.remove_edge(u, v)
        # ``sheaf_consistency_score`` returns a value in [0, 1]. A higher
        # score indicates better alignment between the sheaf and hypergraph.
        score = sheaf_consistency_score(g)
        g.add_edge(u, v)
        if score <= s_threshold:
            continue
        delete = f"MATCH (a {{id: '{u}'}})-[r]- (b {{id: '{v}'}}) DELETE r"
        merge = f"MATCH (a {{id: '{u}'}}),(b {{id: '{v}'}}) MERGE (a)-[:RELATED]->(b)"
        suggestions.append(RepairSuggestion(u, v, float(delta), delete, merge))
    return suggestions


def main(
    dataset: str = "demo", top: int = 5, tau: float = 0.0
) -> List[RepairSuggestion]:
    """Generate and store repair suggestions for ``dataset``.

    The function writes serialised suggestions to the ``repair_suggestions``
    list in Redis and returns the list of suggestions.
    """

    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = DatasetBuilder.from_redis(client, f"dataset:{dataset}", driver)
    g = ds.graph if not callable(ds.graph) else ds.graph()
    suggestions = collect_incoherent_edges(g, top=top, tau=tau)
    if client is not None:
        client.delete("repair_suggestions")
        for s in suggestions:
            client.rpush("repair_suggestions", s.to_json())
    if driver is not None:
        driver.close()
    return suggestions


if __name__ == "__main__":  # pragma: no cover - manual invocation
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="demo", help="Dataset name")
    parser.add_argument("--top", type=int, default=5, help="Number of edges")
    parser.add_argument("--tau", type=float, default=0.0, help="Delta lambda threshold")
    args = parser.parse_args()
    out = main(args.dataset, top=args.top, tau=args.tau)
    print(f"Stored {len(out)} suggestions in repair_suggestions")
