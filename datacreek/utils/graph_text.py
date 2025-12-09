from __future__ import annotations

"""Utilities for converting graph neighborhoods to text."""

from typing import Iterable, List

import networkx as nx


def neighborhood_to_sentence(graph: nx.Graph, path: Iterable, depth: int = 0) -> str:
    """Return a textual description for ``path`` including neighbor context.

    The algorithm recursively summarizes neighbors around each node up to
    ``depth``. Relations connecting nodes in ``path`` are inserted between the
    main elements so the resulting sentence can be fed to a language model for
    fine-tuning.
    """

    nodes = list(path)
    if not nodes:
        return ""

    visited = set(nodes)

    def describe(node: str, lvl: int) -> str:
        """Return text for ``node`` plus recursively described neighbors."""

        text = str(graph.nodes[node].get("text", node))
        if lvl <= 0:
            return text

        parts = []
        for nb in graph.neighbors(node):
            if nb in visited:
                continue
            visited.add(nb)
            rel = graph[node][nb].get("relation", "related to")
            parts.append(f"{rel} {describe(nb, lvl - 1)}")
        if parts:
            text += " (" + ", ".join(parts) + ")"
        return text

    parts: List[str] = []
    for u, v in zip(nodes[:-1], nodes[1:]):
        parts.append(describe(u, depth))
        rel = graph[u][v].get("relation", "->") if graph.has_edge(u, v) else "->"
        parts.append(rel)
    parts.append(describe(nodes[-1], depth))
    sent = " ".join(parts)
    if not sent.endswith("."):
        sent += "."
    return sent


def subgraph_to_text(graph: nx.Graph, nodes: Iterable) -> str:
    """Return a short textual summary of a subgraph.

    Parameters
    ----------
    graph:
        Graph containing the subgraph to describe.
    nodes:
        Iterable of node identifiers making up the subgraph.

    Returns
    -------
    str
        Human readable summary listing edges with their relations.
    """

    sub = graph.subgraph(nodes)
    lines: List[str] = []
    for u, v, data in sub.edges(data=True):
        u_text = str(graph.nodes[u].get("text", u))
        v_text = str(graph.nodes[v].get("text", v))
        rel = data.get("relation", "->")
        lines.append(f"{u_text} {rel} {v_text}")
    summary = ". ".join(lines)
    if summary and not summary.endswith("."):
        summary += "."
    return summary


def graph_to_text(graph: nx.Graph) -> str:
    """Return a summary listing every edge relation in ``graph``."""

    lines: List[str] = []
    for u, v, data in graph.edges(data=True):
        u_text = str(graph.nodes[u].get("text", u))
        v_text = str(graph.nodes[v].get("text", v))
        rel = data.get("relation", "->")
        lines.append(f"{u_text} {rel} {v_text}")
    summary = ". ".join(lines)
    if summary and not summary.endswith("."):
        summary += "."
    return summary
