"""Common pytest fixtures for stopping background services."""

from __future__ import annotations

import asyncio
import logging

import pytest

# Imports moved to fixture to avoid circular dependencies during collection

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def cleanup_background_resources():
    """Ensure long-running helpers stop between tests."""

    yield

    try:
        from datacreek.utils.cache import stop_ttl_manager
        stop_ttl_manager()
    except Exception as exc:
        logger.debug("stopping TTL manager failed: %s", exc)
    try:
        from datacreek.core.knowledge_graph import stop_cleanup_watcher
        stop_cleanup_watcher()
    except Exception as exc:
        logger.debug("stopping knowledge graph watcher failed: %s", exc)
