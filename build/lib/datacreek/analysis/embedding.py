"""Embedding utilities with wall-clock usage Prometheus metrics.

This module exposes a context manager recording the wall-clock seconds spent
when computing embeddings for a given tenant.  The collected data feeds billing
dashboards that combine CPU and GPU costs, while exposing a registry-friendly
API for instrumentation.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator, Dict

from prometheus_client import CollectorRegistry, Counter, REGISTRY

from metrics_prometheus import CpuCostTracker

_EMBEDDING_COUNTERS: Dict[int, Counter] = {}


def _registry_key(registry: CollectorRegistry | None) -> int:
    resolved = registry or REGISTRY
    return id(resolved)


def _create_embedding_counter(registry: CollectorRegistry | None) -> Counter:
    resolved = registry or REGISTRY
    key = _registry_key(registry)
    counter = _EMBEDDING_COUNTERS.get(key)
    if counter is None:
        counter = Counter(
            "embedding_wall_seconds_total",
            "Wall-clock seconds spent computing embeddings per tenant",
            ["tenant"],
            registry=resolved,
        )
        _EMBEDDING_COUNTERS[key] = counter
    return counter


def get_embedding_wall_seconds_counter(
    registry: CollectorRegistry | None = None,
) -> Counter:
    """Return or create the counter bound to ``registry``."""

    if registry is None:
        return embedding_wall_seconds_total
    return _create_embedding_counter(registry)


# Counter tracking wall-clock seconds consumed by embeddings per tenant.
# The metric follows the ``*_total`` naming convention so Prometheus exposes
# ``embedding_wall_seconds_total`` as the published sample.
embedding_wall_seconds_total = _create_embedding_counter(REGISTRY)
# Maintain the legacy name for backwards compatibility.
embedding_cpu_seconds_total = embedding_wall_seconds_total


@contextmanager
def track_embedding_cpu_seconds(
    tenant: str,
    tracker: CpuCostTracker | None = None,
    registry: CollectorRegistry | None = None,
) -> Iterator[None]:
    """Record CPU time spent computing embeddings for ``tenant``.

    Parameters
    ----------
    tenant:
        Tenant identifier associated with the embedding job.
    tracker:
        Optional :class:`~metrics_prometheus.cpu_billing.CpuCostTracker` used to
        translate the measured time into monetary cost.  When ``None`` (the
        default) only the time counter is updated.
    registry:
        Optional Prometheus registry that houses the counter.  When left ``None``
        the module-level counter ``embedding_wall_seconds_total`` is reused.

    Notes
    -----
    The timer relies on :func:`time.perf_counter` which measures wall-clock time
    with high resolution.  The recorded value is added to the
    :data:`embedding_wall_seconds_total` counter (the legacy alias
    :data:`embedding_cpu_seconds_total`) which can be joined with GPU metrics to
    produce unified billing reports.
    """

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        counter = get_embedding_wall_seconds_counter(registry)
        counter.labels(tenant=tenant).inc(elapsed)
        if tracker is not None:
            tracker.record(tenant, elapsed)
