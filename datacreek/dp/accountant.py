"""Renyi differential privacy accountant.

Implements the moments accountant technique from Mironov 2017.
"""

from __future__ import annotations

from math import exp, log
from typing import Iterable, Sequence

_DEFAULT_ALPHAS = tuple(range(2, 33))


def renyi_epsilon(
    epsilons: Sequence[float], *, alphas: Iterable[float] | None = None
) -> float:
    """Return cumulative epsilon using Renyi accounting.

    Parameters
    ----------
    epsilons:
        Sequence of per-query privacy costs ``\varepsilon_i``.
    alphas:
        Candidate Renyi orders. Defaults to ``2..32``.
    """
    if alphas is None:
        alphas = _DEFAULT_ALPHAS
    events = [float(e) for e in epsilons]
    best = float("inf")
    for a in alphas:
        if a <= 1:
            continue
        total = sum(exp((a - 1) * e) for e in events)
        eps = log(total) / (a - 1)
        if eps < best:
            best = eps
    return best


def compute_epsilon(
    epsilons: Sequence[float], *, alphas: Iterable[float] | None = None
) -> float:
    """Return Renyi epsilon given a sequence of privacy events.

    Parameters
    ----------
    epsilons:
        Sequence of per-request costs ``\varepsilon_i``.
    alphas:
        Candidate Renyi orders. Defaults to ``2..32``.
    """

    return renyi_epsilon(epsilons, alphas=alphas)


def allow_request(
    epsilons: Sequence[float],
    epsilon_max: float,
    *,
    alphas: Iterable[float] | None = None,
) -> bool:
    """Return ``True`` if the cumulative epsilon does not exceed ``epsilon_max``."""
    return renyi_epsilon(epsilons, alphas=alphas) <= epsilon_max
