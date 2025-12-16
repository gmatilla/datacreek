"""Core components for datacreek.

This module avoids importing heavy dependencies at module import time by
loading objects lazily via ``__getattr__``.  Only when ``AppContext`` is
explicitly requested will the full context machinery (and its transitive
dependencies) be imported.
"""

__all__ = ["AppContext"]


def __getattr__(name: str):
    if name == "AppContext":
        from .context import AppContext as _AppContext

        return _AppContext
    raise AttributeError(name)
