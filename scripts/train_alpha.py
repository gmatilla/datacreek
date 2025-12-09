"""Meta-optimise multiplex weights ``alpha_t`` on validation Macro-F1.

The script provides a :func:`train_alpha` helper implementing a projected
gradient ascent step on the simplex.  External callers are expected to provide a
``grad_fn`` that returns the gradient of the validation Macro-F1 with respect to
``alpha``.  The routine keeps the weights non-negative and normalised so that

.. math::

   \sum_t \alpha_t = 1.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def train_alpha(
    alpha_init: np.ndarray,
    grad_fn: Callable[[np.ndarray], np.ndarray],
    *,
    lr: float = 0.1,
    steps: int = 100,
) -> np.ndarray:
    """Optimise ``alpha`` via projected gradient ascent.

    Parameters
    ----------
    alpha_init:
        Initial weights for each edge type.  They will be projected onto the
        probability simplex.
    grad_fn:
        Callable returning the gradient of validation Macro-F1 with respect to
        ``alpha`` at the current point.
    lr:
        Step size for the gradient ascent update.
    steps:
        Number of gradient steps to perform.

    Returns
    -------
    numpy.ndarray
        Optimised weights that satisfy the constraints ``alpha_t >= 0`` and
        ``sum(alpha) == 1``.
    """

    alpha = np.asarray(alpha_init, dtype=float)
    alpha = np.clip(alpha, 0, None)
    alpha /= alpha.sum()

    for _ in range(steps):
        grad = grad_fn(alpha)
        alpha = alpha + lr * grad
        alpha = np.clip(alpha, 0, None)
        alpha_sum = alpha.sum()
        if alpha_sum == 0:
            # Avoid division by zero; fall back to uniform distribution.
            alpha = np.full_like(alpha, 1.0 / len(alpha))
        else:
            alpha /= alpha_sum
    return alpha
