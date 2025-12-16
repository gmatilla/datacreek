"""Curriculum learning dataloader ordered by sample difficulty.

This module provides utilities to compute a difficulty score for each
sample based on hypergraph metrics and to iterate over the dataset from
simplest to most complex examples. The difficulty of a sample *d* is
defined as:

.. math::

    d = \gamma h + \delta l + \eta c,

where ``h`` is the number of hops in the hypergraph, ``l`` the prompt
length and ``c`` the centrality of the target node. The weights
``\gamma``, ``\delta`` and ``\eta`` default respectively to ``0.5``,
``0.3`` and ``0.2`` but can be adjusted when constructing the
:class:`CurriculumDataLoader`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence


def compute_difficulty(
    sample: Dict[str, object],
    gamma: float = 0.5,
    delta: float = 0.3,
    eta: float = 0.2,
) -> float:
    """Compute the curriculum difficulty for a single sample.

    Parameters
    ----------
    sample:
        Mapping describing the sample. Expected keys are ``hops``,
        ``prompt`` and ``centrality`` but missing keys default to ``0``.
    gamma, delta, eta:
        Weights associated with the number of hops, prompt length and
        centrality respectively.

    Returns
    -------
    float
        The difficulty score ``d``.
    """

    h = float(sample.get("hops", 0))
    l = float(len(str(sample.get("prompt", ""))))
    c = float(sample.get("centrality", 0))
    return gamma * h + delta * l + eta * c


@dataclass
class CurriculumDataLoader:
    """Iterate over samples sorted by curriculum difficulty.

    Parameters
    ----------
    dataset:
        Sequence of samples to draw from.
    batch_size:
        Number of samples per batch.
    gamma, delta, eta:
        Weights forwarded to :func:`compute_difficulty`.
    """

    dataset: Sequence[Dict[str, object]]
    batch_size: int
    gamma: float = 0.5
    delta: float = 0.3
    eta: float = 0.2

    def __post_init__(self) -> None:
        # Pre-compute sorted dataset to ensure deterministic ordering.
        self._sorted = sorted(
            self.dataset,
            key=lambda s: compute_difficulty(s, self.gamma, self.delta, self.eta),
        )

    def __iter__(self) -> Iterator[List[Dict[str, object]]]:
        """Yield batches from easiest to hardest samples."""

        for i in range(0, len(self._sorted), self.batch_size):
            yield list(self._sorted[i : i + self.batch_size])

    def __len__(self) -> int:  # pragma: no cover - trivial
        return (len(self._sorted) + self.batch_size - 1) // self.batch_size
