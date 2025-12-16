"""Stratified reservoir sampler maintaining class quotas.

This module provides :class:`StratifiedReservoirSampler` which performs
reservoir sampling in a streaming setting while keeping the class
composition close to a requested distribution.  For each class ``c`` a
reservoir of size ``k_c`` is allocated based on the desired ratio
``r_c`` and the overall capacity ``k`` (default 10k).  Incoming samples
are stored or replace existing ones using the classic algorithm of
[Vitter 1985] to ensure each observation has an equal chance of being
retained within its class quota.

Variables
---------
``k``
    Total capacity of the global reservoir.
``ratios``
    Mapping from class label to target fraction of the dataset
    (values must sum to 1).
``quotas``
    Integer number of slots reserved for each class ``k_c``.

"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Dict, List, MutableMapping, Sequence


class StratifiedReservoirSampler:
    """Reservoir sampler that enforces per-class quotas.

    Parameters
    ----------
    k:
        Total number of samples to keep across all classes.  Default is
        ``10_000``.
    class_ratios:
        Mapping of ``label -> desired_fraction``.  Fractions must sum to
        one; quotas are computed as ``k * fraction``.
    random_state:
        Optional seed or :class:`random.Random` instance for
        reproducibility.
    """

    def __init__(
        self,
        k: int = 10_000,
        class_ratios: (
            MutableMapping[Any, float] | Sequence[tuple[Any, float]] | None
        ) = None,
        random_state: int | random.Random | None = None,
    ) -> None:
        if class_ratios is None:
            raise ValueError("class_ratios must be provided")

        if isinstance(class_ratios, MutableMapping):
            ratios = dict(class_ratios)
        else:
            ratios = dict(class_ratios)

        total = sum(ratios.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError("class_ratios must sum to 1")

        self.k = int(k)
        self.ratios = ratios
        # Use the standard pseudo-random generator for deterministic behavior
        # when a ``random_state`` is provided. The sampler is not used for
        # cryptographic purposes, so Bandit's B311 warning is suppressed.
        self._rng = random.Random(random_state)  # nosec B311

        # Determine integer quotas per class while ensuring the sum is exactly k
        float_quota = {c: r * self.k for c, r in self.ratios.items()}
        quotas = {c: int(math.floor(v)) for c, v in float_quota.items()}
        remaining = self.k - sum(quotas.values())
        # Distribute leftover capacity to classes with largest fractional part
        for c, _ in sorted(
            float_quota.items(),
            key=lambda item: item[1] - quotas[item[0]],
            reverse=True,
        )[:remaining]:
            quotas[c] += 1
        self.quotas: Dict[Any, int] = quotas

        self._reservoirs: Dict[Any, List[Any]] = {c: [] for c in ratios}
        self._seen: Dict[Any, int] = defaultdict(int)

    def add(self, item: Any, label: Any) -> None:
        """Add a sample with its class label.

        Uses reservoir sampling so that after processing ``n`` items of a
        given class each has probability ``min(1, k_c / n)`` of residing
        in the final sample.
        """

        if label not in self.quotas:
            raise KeyError(f"Unknown class {label!r}")

        self._seen[label] += 1
        quota = self.quotas[label]
        reservoir = self._reservoirs[label]

        if len(reservoir) < quota:
            reservoir.append(item)
            return

        j = self._rng.randrange(self._seen[label])
        if j < quota:
            reservoir[j] = item

    def samples(self) -> List[Any]:
        """Return the concatenated list of samples from all classes."""

        result: List[Any] = []
        for res in self._reservoirs.values():
            result.extend(res)
        return result

    def stats(self) -> Dict[str, int]:
        """Return counts of retained samples per class."""

        return {c: len(res) for c, res in self._reservoirs.items()}


__all__ = ["StratifiedReservoirSampler"]
