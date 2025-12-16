"""Graph-based reward utilities for RLHF.

This module extracts simple (subject, predicate, object) triplets from a model
response and verifies them against a provided hypergraph. The resulting ratio of
verified triplets can be plugged into ``ppo_config.reward_fn`` to discourage
hallucinations during reinforcement learning from human feedback.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, Iterable, List, Set, Tuple

# Type aliases for clarity
Triplet = Tuple[str, str, str]
HyperGraph = Dict[Tuple[str, str], Set[str]]

# Heuristic regex capturing "subject predicate object" patterns with a restricted
# verb vocabulary. Using underscores allows multi-word relations such as
# ``capital_of``.
_TRIPLET_RE = re.compile(
    r"([A-Za-z_]+)\s+(is|likes|has|eats|knows|located_in|capital_of)\s+([A-Za-z_]+)",
    flags=re.IGNORECASE,
)


def extract_triplets(text: str) -> List[Triplet]:
    """Extract knowledge triplets from ``text``.

    Parameters
    ----------
    text:
        Model response containing statements like ``"Paris located_in France"``.

    Returns
    -------
    list of tuple
        List of ``(subject, predicate, object)`` triplets, lowercased for
        consistent matching.
    """
    return [
        (m.group(1).lower(), m.group(2).lower(), m.group(3).lower())
        for m in _TRIPLET_RE.finditer(text)
    ]


def verify_triplets(triplets: Iterable[Triplet], graph: HyperGraph) -> float:
    """Compute fraction of ``triplets`` present in ``graph``.

    The hypergraph is represented as a mapping from ``(subject, predicate)``
    pairs to a set of valid objects.

    Parameters
    ----------
    triplets:
        Iterable of triplets to verify.
    graph:
        Hypergraph containing factual relations.

    Returns
    -------
    float
        Ratio of verified triplets. Returns ``0.0`` when no triplets are
        provided.
    """
    triplet_list = list(triplets)
    if not triplet_list:
        return 0.0
    valid = sum(1 for (s, p, o) in triplet_list if o in graph.get((s, p), set()))
    return valid / len(triplet_list)


def build_reward_fn(graph: HyperGraph) -> Callable[[str], float]:
    """Create a reward function tied to ``graph``.

    The returned callable extracts triplets from a model response and returns the
    verification ratio against ``graph``. It can be assigned to
    ``ppo_config.reward_fn`` for PPO training.

    Examples
    --------
    >>> graph = {("paris", "located_in"): {"france"}}
    >>> reward_fn = build_reward_fn(graph)
    >>> reward_fn("Paris located_in France")
    1.0
    """

    def reward_fn(response: str) -> float:
        triplets = extract_triplets(response)
        return verify_triplets(triplets, graph)

    return reward_fn


__all__ = ["extract_triplets", "build_reward_fn", "verify_triplets"]
