"""Alias-aware fact-check reward utilities.

This module extends the basic graph-based reward function by resolving
entity aliases through a Neo4j full-text index. The reward is computed as

.. math::

    R = \frac{N_\text{facts} + N_\text{alias}}{N_\text{facts}}

where :math:`N_\text{facts}` counts exact matches between extracted
triplets and the knowledge graph while :math:`N_\text{alias}` counts
matches found after alias canonicalisation via ``apoc.text.clean``.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Set, Tuple

from neo4j import Driver

from .auto_feedback import extract_triplets

# Type aliases mirroring ``auto_feedback`` for clarity.
Triplet = Tuple[str, str, str]
HyperGraph = Dict[Tuple[str, str], Set[str]]


def _resolve_alias(driver: Driver | None, name: str) -> Set[str]:
    """Return canonical entity names for ``name`` using Neo4j aliases.

    Parameters
    ----------
    driver:
        Active Neo4j driver. When ``None`` the function falls back to the
        identity mapping which keeps the entity unchanged. This allows the
        reward function to operate without Neo4j during tests.
    name:
        Raw entity name possibly containing punctuation or casing
        variations.

    Returns
    -------
    set of str
        Set of canonical names obtained through ``apoc.text.clean`` and a
        full-text index named ``alias``. If no mapping is found the original
        name is returned.
    """

    if driver is None:
        return {name}

    query = (
        "WITH apoc.text.clean($n) AS cleaned "
        "CALL db.index.fulltext.queryNodes('alias', cleaned) YIELD node "
        "RETURN node.canonical AS canonical"
    )

    with driver.session() as session:
        records = session.run(query, n=name)
        resolved = {r["canonical"] for r in records}  # type: ignore[index]
    return resolved or {name}


def build_alias_reward_fn(
    graph: HyperGraph, driver: Driver | None
) -> Callable[[str], float]:
    """Create an alias-aware reward function.

    The returned callable extracts triplets from a model ``response`` and
    verifies them against ``graph``. When a triplet is not found directly,
    the subjects and objects are canonicalised through Neo4j to discover
    alias matches. The reward is the ratio of validated facts—either exact
    or via alias—over the number of extracted facts.

    Parameters
    ----------
    graph:
        Knowledge graph mapping ``(subject, predicate)`` pairs to a set of
        accepted objects.
    driver:
        Neo4j driver providing access to the alias index. ``None`` skips alias
        resolution, useful for offline tests.

    Returns
    -------
    callable
        Function taking a text ``response`` and returning the reward in
        ``[0, 1]``.
    """

    def reward_fn(response: str) -> float:
        triplets = extract_triplets(response)
        if not triplets:
            return 0.0

        validated = 0
        alias_validated = 0

        for subj, pred, obj in triplets:
            # Exact match
            objects = graph.get((subj, pred), set())
            if obj in objects:
                validated += 1
                continue

            # Alias-based match
            subj_aliases = _resolve_alias(driver, subj)
            obj_aliases = _resolve_alias(driver, obj)
            if any(
                alias_obj in graph.get((alias_subj, pred), set())
                for alias_subj in subj_aliases
                for alias_obj in obj_aliases
            ):
                alias_validated += 1

        return (validated + alias_validated) / len(triplets)

    return reward_fn


__all__ = ["build_alias_reward_fn", "_resolve_alias"]
