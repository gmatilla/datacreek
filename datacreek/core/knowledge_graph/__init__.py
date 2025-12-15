"""Knowledge Graph core package."""
from __future__ import annotations

from .base import KnowledgeGraph
from .watchdog import (
    CleanupConfig,
    apply_cleanup_config,
    get_cleanup_cfg,
    start_cleanup_watcher,
    stop_cleanup_watcher,
    verify_thresholds,
)

__all__ = [
    "KnowledgeGraph",
    "CleanupConfig",
    "apply_cleanup_config",
    "get_cleanup_cfg",
    "start_cleanup_watcher",
    "stop_cleanup_watcher",
    "verify_thresholds",
]
