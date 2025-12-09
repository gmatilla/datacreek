"""Utilities for soft deletion of nodes and embedding vectors.

Each entity exposes a ``deleted_at`` timestamp, allowing records to be
flagged as removed without immediate physical deletion. A periodic purge
can later remove entries older than the retention window.

The module defines simple ``Node`` and ``Vector`` dataclasses for test
purposes and provides two helper functions:

``mark_deleted(entity)``
    Set ``deleted_at`` on the entity to the current UTC time.

``purge_deleted(collection, now=None, days=30)``
    Remove entities from ``collection`` whose ``deleted_at`` is older than
    ``days`` from ``now``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Protocol, TypeVar


class HasDeletedAt(Protocol):
    """Protocol describing an entity with an optional ``deleted_at`` field."""

    deleted_at: datetime | None


T = TypeVar("T", bound=HasDeletedAt)


@dataclass
class Node:
    """Neo4j node representation with soft-delete metadata."""

    id: str
    deleted_at: datetime | None = field(default=None, repr=False)


@dataclass
class Vector:
    """Embedding vector record with soft-delete metadata."""

    id: str
    values: List[float] = field(default_factory=list)
    deleted_at: datetime | None = field(default=None, repr=False)


def mark_deleted(entity: HasDeletedAt) -> None:
    """Flag ``entity`` as deleted by setting ``deleted_at`` to ``datetime.utcnow()``.

    Parameters
    ----------
    entity:
        Any object exposing a ``deleted_at`` attribute.
    """

    entity.deleted_at = datetime.now(timezone.utc)


def purge_deleted(
    collection: Iterable[T], now: datetime | None = None, *, days: int = 30
) -> List[T]:
    """Return entities not scheduled for deletion and drop expired entries.

    Parameters
    ----------
    collection:
        Iterable of entities implementing :class:`HasDeletedAt`.
    now:
        Reference time for computing the retention window. Defaults to ``datetime.utcnow``.
    days:
        Retention period in days. Entities with ``deleted_at`` older than this
        delta will be purged.

    Returns
    -------
    List[T]
        The list of entities that remain after purging.
    """

    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    return [e for e in collection if not e.deleted_at or e.deleted_at > cutoff]
