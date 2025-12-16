"""Array API selection utilities.

This module exposes :func:`get_xp` which returns either the ``cupy`` or
``numpy`` module based on GPU availability. It allows analysis code to use
``xp``-prefixed operations with transparent CPU/GPU support.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any


def _cupy_available() -> bool:
    """Return ``True`` if :mod:`cupy` is installed and a GPU is detected."""
    try:
        cp = import_module("cupy")
        cuda = getattr(cp, "cuda", None)
        if cuda is None:
            return False
        return bool(cuda.runtime.getDeviceCount() > 0)
    except Exception:
        return False


def get_xp(obj: Any | None = None) -> ModuleType:
    """Return array API module ``cupy`` or ``numpy``.

    Parameters
    ----------
    obj:
        Optional array-like used to infer the appropriate module.
        When ``obj`` is already a ``cupy`` array, :mod:`cupy` is returned.
        GPU checks are skipped, helping when arrays are created externally.
    """
    if obj is not None and obj.__class__.__module__.startswith("cupy"):
        try:
            return import_module("cupy")
        except Exception:
            return import_module("numpy")

    if _cupy_available():
        return import_module("cupy")

    return import_module("numpy")
