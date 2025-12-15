"""Helper utilities for importing optional heavy dependencies."""

from __future__ import annotations

import importlib
import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


def optional_import(module_name: str, attr: str | None = None) -> Any | None:
    """Import a module or attribute, returning ``None`` if unavailable."""

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        LOGGER.debug("Optional dependency %s unavailable: %s", module_name, exc)
        return None
    if attr is None:
        return module
    return getattr(module, attr, None)
