from itertools import islice
from typing import Dict, Iterable, List, Set, Tuple

import networkx as nx


def automorphisms(graph: nx.Graph, max_count: int = 10) -> List[Dict[object, object]]:
    """Return up to ``max_count`` automorphisms of ``graph``.

    Each automorphism is represented as a mapping ``node -> node``.
    The function relies on :class:`~networkx.algorithms.isomorphism.GraphMatcher`
    which may be costly for large graphs, so ``max_count`` limits the number of
    mappings enumerated.
    """
    matcher = nx.algorithms.isomorphism.GraphMatcher(graph, graph)
    return list(islice(matcher.isomorphisms_iter(), max_count))


def automorphism_orbits(graph: nx.Graph, max_count: int = 10) -> List[Set[object]]:
    """Return node orbits induced by automorphisms.

    Parameters
    ----------
    graph:
        Input graph.
    max_count:
        Maximum number of automorphisms explored.

    Returns
    -------
    list[set]
        Each set contains nodes that can be permuted onto each other by some
        automorphism.
    """
    autos = automorphisms(graph, max_count=max_count)
    orbits: List[Set[object]] = []
    for mapping in autos:
        for a, b in mapping.items():
            if a == b:
                continue
            found = False
            for orb in orbits:
                if a in orb or b in orb:
                    orb.update({a, b})
                    found = True
                    break
            if not found:
                orbits.append({a, b})
    return orbits


def automorphism_group_order(graph: nx.Graph, max_count: int = 100) -> int:
    """Return the size of the automorphism group.

    Parameters
    ----------
    graph:
        Input graph.
    max_count:
        Maximum number of distinct automorphisms to enumerate. The method
        stops early once ``max_count`` mappings are found and returns this
        value as a lower bound on the true group order.

    Notes
    -----
    The enumeration relies on :class:`~networkx.algorithms.isomorphism.GraphMatcher`.
    For large graphs only a subset of mappings may be explored which yields a
    lower bound on the actual automorphism group size.
    """

    matcher = nx.algorithms.isomorphism.GraphMatcher(graph, graph)
    count = 0
    for _ in islice(matcher.isomorphisms_iter(), max_count):
        count += 1
        if count >= max_count:
            break
    return count


def quotient_graph(
    graph: nx.Graph, orbits: Iterable[Set[object]]
) -> Tuple[nx.Graph, Dict[object, int]]:
    """Return graph quotienting nodes by ``orbits``.

    Parameters
    ----------
    graph:
        Input graph.
    orbits:
        Collection of node sets representing equivalence classes.

    Returns
    -------
    tuple[nx.Graph, dict]
        The quotient graph and a mapping ``node -> class_id``.
    """
    mapping: Dict[object, int] = {}
    for idx, orbit in enumerate(orbits):
        for n in orbit:
            mapping[n] = idx
    counter = len(orbits)
    for n in graph.nodes():
        if n not in mapping:
            mapping[n] = counter
            counter += 1
    q = nx.Graph()
    for u, v in graph.edges():
        a, b = mapping[u], mapping[v]
        if a == b:
            continue
        q.add_edge(a, b)
    return q, mapping
