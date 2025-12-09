"""Embedding utilities with CPU usage Prometheus metrics.

This module exposes a context manager recording the CPU seconds spent
when computing embeddings for a given tenant.  The collected data feeds
billing dashboards that combine CPU and GPU costs.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from prometheus_client import Counter

from metrics_prometheus import CpuCostTracker

# Counter tracking CPU seconds consumed by embedding computation per tenant.
# The metric follows the ``*_total`` naming convention so Prometheus exposes
# ``embedding_cpu_seconds_total_total`` as the sample name.
embedding_cpu_seconds_total = Counter(
    "embedding_cpu_seconds_total",
    "CPU seconds spent computing embeddings per tenant",
    ["tenant"],
)


@contextmanager
def track_embedding_cpu_seconds(
    tenant: str, tracker: CpuCostTracker | None = None
) -> Iterator[None]:
    """Record CPU time spent computing embeddings for ``tenant``.

    Parameters
    ----------
    tenant:
        Tenant identifier associated with the embedding job.
    tracker:
        Optional :class:`~metrics_prometheus.cpu_billing.CpuCostTracker` used to
        translate the measured CPU time into monetary cost.  When ``None`` (the
        default) only the time counter is updated.

    Notes
    -----
    The timer relies on :func:`time.perf_counter` which measures wall-clock
    time with high resolution.  The recorded value is added to the
    :data:`embedding_cpu_seconds_total` counter which can be joined with GPU
    metrics to produce unified billing reports.
    """

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        embedding_cpu_seconds_total.labels(tenant=tenant).inc(elapsed)
        if tracker is not None:
            tracker.record(tenant, elapsed)
