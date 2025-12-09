"""Back-pressure controls for ingestion queue."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time

from datacreek.analysis import monitoring

__all__ = [
    "has_capacity",
    "acquire_slot",
    "acquire_slot_with_backoff",
    "release_slot",
    "set_limit",
    "active_count",
]

_limit = int(os.getenv("INGEST_QUEUE_LIMIT", "10000"))
_queue: asyncio.Queue[None] = asyncio.Queue(maxsize=_limit)
_lock = threading.Lock()


def _update_metric() -> None:
    """Record current queue usage in Prometheus."""
    try:
        monitoring.update_metric(
            "ingest_queue_fill_ratio",
            _queue.qsize() / _limit,
        )
    except Exception:  # pragma: no cover - metrics optional
        pass


def set_limit(limit: int) -> None:
    """Set queue limit (testing only)."""
    global _limit, _queue
    with _lock:
        _limit = limit
        _queue = asyncio.Queue(maxsize=limit)
    _update_metric()


def has_capacity() -> bool:
    """Return ``True`` if the queue is below its limit."""
    with _lock:
        return _queue.qsize() < _limit


def acquire_slot() -> bool:
    """Try to reserve one slot for an ingest task."""
    with _lock:
        try:
            _queue.put_nowait(None)
            success = True
        except asyncio.QueueFull:
            success = False
    _update_metric()
    return success


def acquire_slot_with_backoff(
    retries: int = 3,
    base_delay: float = 0.05,
    *,
    spool_dir: str | None = None,
    spool_data: dict | None = None,
) -> bool:
    """Try to acquire a slot with exponential backoff.

    Parameters
    ----------
    retries:
        Number of additional attempts when the queue is full.
    base_delay:
        Initial delay in seconds for backoff.
    spool_dir:
        Optional directory where dropped tasks are recorded when all
        attempts fail.
    spool_data:
        Optional JSON-serializable payload describing the task.
    """

    for attempt in range(retries + 1):
        if acquire_slot():
            return True
        delay = base_delay * (2**attempt)
        time.sleep(delay)

    if spool_dir:
        os.makedirs(spool_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        path = os.path.join(spool_dir, f"spool_{ts}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(spool_data or {}, f)
        except Exception:
            pass
    return False


def release_slot() -> None:
    """Release a previously acquired slot."""
    with _lock:
        if not _queue.empty():
            _queue.get_nowait()
    _update_metric()


def active_count() -> int:
    """Return the number of active ingest tasks."""
    with _lock:
        return _queue.qsize()
