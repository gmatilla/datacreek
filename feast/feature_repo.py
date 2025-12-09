"""Feature store definitions and cache utilities for embedding vectors.

This module configures a minimal Feast repository with a single entity and
feature view.  The entity ``embedding_hash`` uniquely identifies an embedding
vector.  The feature view ``vector_fp8`` stores a quantised FP8 representation
of the vector.  A tiny in-memory cache is provided to showcase how repeated
lookups for the same hash can be served without hitting the online store again.

Mathematically, an embedding \(v\) is stored as an array of low precision
floating point numbers \(v \in \mathbb{R}^n\).  When a hash \(h\) is queried
multiple times, the cache hit rate is defined as:

.. math::
   \text{hit\_rate} = \frac{N_{\text{hits}}}{N_{\text{lookups}}}

where :math:`N_{\text{hits}}` is the number of times a query is resolved from
cache and :math:`N_{\text{lookups}}` the total number of requests.
"""

from __future__ import annotations

from typing import Any, Dict, List


class Entity:
    """Minimal stand‑in for :class:`feast.entity.Entity`.

    Only the ``name`` and ``description`` fields are required for the unit
    tests and documentation examples; additional Feast behaviour is outside the
    scope of this lightweight stub.
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description


class Field:
    """Simplified schema field used by :class:`FeatureView` declarations."""

    def __init__(self, name: str, dtype: Any) -> None:
        self.name = name
        self.dtype = dtype


class Array:
    """Placeholder type representing a Feast ``Array`` wrapper."""

    def __init__(self, dtype: Any) -> None:
        self.dtype = dtype


class Float32:
    """Marker type standing in for ``feast.types.Float32``."""


class FileSource:
    """Trivial representation of an offline file source."""

    def __init__(self, path: str, timestamp_field: str) -> None:
        self.path = path
        self.timestamp_field = timestamp_field


class FeatureView:
    """Container describing how to access feature data for an entity."""

    def __init__(
        self, name: str, entities: List[Entity], schema: List[Field], source: FileSource
    ) -> None:
        self.name = name
        self.entities = entities
        self.schema = schema
        self.source = source


class FeatureStore:  # pragma: no cover - behaviour mocked in tests
    """Placeholder feature store used purely for type hints."""

    def get_online_features(self, *args: Any, **kwargs: Any):  # noqa: D401
        """Retrieve features from the online store."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Feast object declarations
# ---------------------------------------------------------------------------

embedding = Entity(
    name="embedding_hash",
    description="Unique 64-bit hash identifying an embedding vector.",
)
"""Feast entity identifying each embedding by a stable hash."""

vector_source = FileSource(
    path="dataset/feature_repo/vector_fp8.parquet",
    timestamp_field="event_timestamp",
)
"""Placeholder offline source; in production this points to a materialised table."""

vector_fp8_view = FeatureView(
    name="vector_fp8",
    entities=[embedding],
    schema=[Field(name="vector_fp8", dtype=Array(Float32))],
    source=vector_source,
)
"""Feature view exposing the FP8-quantised embedding vectors."""

# ---------------------------------------------------------------------------
# Simple in-memory caching layer
# ---------------------------------------------------------------------------

_cache: Dict[str, List[float]] = {}
_hits: int = 0
_lookups: int = 0


def get_vector_fp8(store: FeatureStore, embedding_hash: str) -> List[float]:
    """Return the FP8 vector for ``embedding_hash`` using an in-memory cache.

    Parameters
    ----------
    store:
        A configured ``FeatureStore`` instance.
    embedding_hash:
        Hash that identifies the embedding.

    Returns
    -------
    list of float
        The embedding vector stored in FP8 format.
    """
    global _hits, _lookups
    _lookups += 1
    if embedding_hash in _cache:
        _hits += 1
        return _cache[embedding_hash]

    # Fetch from the online store and update the cache
    response = store.get_online_features(
        features=["vector_fp8:vector_fp8"],
        entity_rows=[{"embedding_hash": embedding_hash}],
    )
    vector = response.to_dict()["vector_fp8"][0]
    _cache[embedding_hash] = vector
    return vector


def cache_hit_rate() -> float:
    """Compute the cache hit rate ``N_hits / N_lookups``.

    Returns
    -------
    float
        Ratio of cache hits to total lookups.  Zero if no lookups were made.
    """
    return _hits / _lookups if _lookups else 0.0


def reset_cache() -> None:
    """Clear the cache and reset hit/lookup counters.

    This helper is primarily intended for unit tests.
    """
    global _hits, _lookups
    _cache.clear()
    _hits = 0
    _lookups = 0
