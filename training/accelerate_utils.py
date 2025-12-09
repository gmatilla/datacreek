"""Utilities for configuring Hugging Face Accelerate.

Includes helpers to compute gradient accumulation steps based on
available GPU memory and to load the shared ``accelerate_config.yaml``
file.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any, Dict

import yaml


def compute_gradient_accumulation_steps(
    batch_size: int, available_memory_gb: float
) -> int:
    """Return gradient accumulation steps using :math:`ceil(batch/avail\_mem)`.

    Parameters
    ----------
    batch_size:
        The global batch size across all workers.
    available_memory_gb:
        Amount of GPU memory available in gigabytes.

    Returns
    -------
    int
        Number of steps to accumulate gradients before an optimizer step.

    Raises
    ------
    ValueError
        If ``batch_size`` or ``available_memory_gb`` is not positive.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if available_memory_gb <= 0:
        raise ValueError("available_memory_gb must be positive")
    return ceil(batch_size / available_memory_gb)


def load_accelerate_config(
    path: str | Path = "accelerate_config.yaml",
) -> Dict[str, Any]:
    """Load the project's accelerate configuration file.

    Parameters
    ----------
    path:
        Optional path to the configuration file.

    Returns
    -------
    Dict[str, Any]
        Parsed YAML configuration as a dictionary.
    """
    with open(Path(path), "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
