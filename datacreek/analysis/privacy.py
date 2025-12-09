"""Privacy utilities including k-out randomized response."""

from __future__ import annotations

import random
from typing import Iterable, List


def k_out_randomized_response(items: Iterable[str], k: int = 2) -> List[str]:
    """Return a list where each element may be replaced by a random one.

    This simple mechanism keeps each item with probability ``k/(k+1)`` and
    replaces it with a uniformly drawn element from ``items`` otherwise.
    It provides a basic k-out randomized response guaranteeing limited
    differential privacy when ``k`` is small (\varepsilon \approx 2 for k=2).
    """
    items = list(items)
    if not items:
        return []
    pool = list(items)
    result: List[str] = []
    for item in items:
        if random.random() < k / (k + 1.0):
            result.append(item)
        else:
            result.append(random.choice(pool))
    return result
