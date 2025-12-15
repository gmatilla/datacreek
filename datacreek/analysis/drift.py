r"""Embedding drift detection utilities.

This module tracks the kernel mean :math:`\mu_0` of baseline validation
embeddings and measures distributional drift after fine-tuning via a simple
Maximum Mean Discrepancy (MMD) metric.

Given a set of new embeddings with mean :math:`\mu_{\text{new}}`, we compute

.. math::

    \text{MMD}^2 = \|\mu_{\text{new}} - \mu_0\|_2^2

The resulting scalar is exported to Prometheus as ``embedding_mmd`` so that
alerts can trigger when drift exceeds a threshold.
"""

from __future__ import annotations

from pathlib import Path
import logging

import numpy as np

from .monitoring import update_metric

LOGGER = logging.getLogger(__name__)


def _ensure_matrix(array: np.ndarray, name: str) -> None:
    if array.ndim != 2:
        msg = f"{name} must be a 2D array; got shape {array.shape}"
        LOGGER.error(msg)
        raise ValueError(msg)
    if array.shape[0] == 0:
        msg = f"{name} must contain at least one sample"
        LOGGER.error(msg)
        raise ValueError(msg)
    if array.shape[1] == 0:
        msg = f"{name} must contain at least one feature"
        LOGGER.error(msg)
        raise ValueError(msg)


def _ensure_vector(array: np.ndarray, name: str) -> None:
    if array.ndim != 1:
        msg = f"{name} must be a 1D vector; got shape {array.shape}"
        LOGGER.error(msg)
        raise ValueError(msg)
    if array.shape[0] == 0:
        msg = f"{name} must contain at least one feature"
        LOGGER.error(msg)
        raise ValueError(msg)


def baseline_mean(embeddings: np.ndarray) -> np.ndarray:
    r"""Return the baseline kernel mean :math:`\mu_0`.

    Parameters
    ----------
    embeddings:
        Array of shape ``(n, d)`` representing baseline validation embeddings.

    Returns
    -------
    np.ndarray
        Mean vector of shape ``(d,)``.
    """

    _ensure_matrix(embeddings, "embeddings")
    return embeddings.mean(axis=0)


def embedding_mmd(new_embeddings: np.ndarray, mu0: np.ndarray) -> float:
    r"""Compute squared MMD between ``new_embeddings`` and ``mu0``.

    Parameters
    ----------
    new_embeddings:
        Array of shape ``(m, d)`` from the latest fine-tuned model.
    mu0:
        Baseline mean vector :math:`\mu_0` from :func:`baseline_mean`.

    Returns
    -------
    float
        Squared MMD value :math:`\|\mu_{\text{new}} - \mu_0\|_2^2`.
    """

    _ensure_matrix(new_embeddings, "new_embeddings")
    _ensure_vector(mu0, "mu0")
    if new_embeddings.shape[1] != mu0.shape[0]:
        msg = (
            "New embeddings and mu0 must share the same feature dimension "
            f"({new_embeddings.shape[1]} != {mu0.shape[0]})"
        )
        LOGGER.error(msg)
        raise ValueError(msg)
    mu_new = new_embeddings.mean(axis=0)
    delta = mu_new - mu0
    mmd2 = float(np.dot(delta, delta))
    update_metric("embedding_mmd", mmd2)
    return mmd2


def save_baseline(mu0: np.ndarray, path: str | Path) -> None:
    """Persist the baseline mean vector ``mu0`` to ``path`` as ``.npy``."""

    np.save(Path(path), mu0)


def load_baseline(path: str | Path) -> np.ndarray:
    """Load a baseline mean vector previously saved with :func:`save_baseline`."""

    return np.load(Path(path))
