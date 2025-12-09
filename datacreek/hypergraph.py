"""Utilities for streaming hypergraph updates.

This module provides two small helpers used to update a hot ``RedisGraph`` layer
from a stream of edge events.  The goal is to approximate the behaviour of a
Flink job with 30 second tumbling windows without pulling in heavy
infrastructure.

The function :func:`process_edge_stream` consumes an iterable of edge events.
Each event must provide a ``ts`` timestamp, a ``src`` node and a ``dst`` node.
Events are grouped into 30 second windows; when a window closes, the edges are
written to the provided hot layer via :meth:`RedisGraphHotLayer.add_edge`.

``RedisGraphHotLayer`` stores edges in memory and tracks the latency of each
operation so that the p95 latency can be monitored.  The latency target from the
checklist is 500 ms:

.. math::
   \text{p95 latency} < 0.5\,\text{s}

The module also exposes a Prometheus counter ``late_edge_total`` that tracks
how many edges arrive after the watermark and are routed to the late-event sink.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

try:  # optional Prometheus metric for late edges
    from prometheus_client import Counter
except Exception:  # pragma: no cover - metrics optional
    Counter = None  # type: ignore

# Prometheus counter incremented when an edge arrives after the watermark
LATE_EDGE_TOTAL: Optional[Counter]
if Counter is not None:
    LATE_EDGE_TOTAL = Counter(
        "late_edge_total",
        "Hypergraph edges routed to the late-event sink",
    )
else:  # pragma: no cover - metrics optional
    LATE_EDGE_TOTAL = None


def process_edge_stream(
    events: Iterable[Dict[str, Any]],
    hot_layer: "RedisGraphHotLayer",
    window: timedelta = timedelta(seconds=30),
) -> None:
    """Group events into ``window`` and write edges incrementally.

    Parameters
    ----------
    events:
        Iterable of event dictionaries with ``src``, ``dst`` and ``ts`` (``datetime``).
    hot_layer:
        Instance of :class:`RedisGraphHotLayer` that receives edge updates.
    window:
        Length of the tumbling window.  The default is 30 seconds as required
        by the specification.
    """
    sorted_events: List[Dict[str, Any]] = sorted(events, key=lambda e: e["ts"])
    batch: List[Dict[str, Any]] = []
    window_start: datetime | None = None

    for event in sorted_events:
        if window_start is None:
            window_start = event["ts"]
        if event["ts"] - window_start < window:
            batch.append(event)
        else:
            _flush_batch(batch, hot_layer)
            batch = [event]
            window_start = event["ts"]
    if batch:
        _flush_batch(batch, hot_layer)


def process_edge_stream_with_watermark(
    events: Iterable[Dict[str, Any]],
    hot_layer: "RedisGraphHotLayer",
    *,
    window: timedelta = timedelta(seconds=30),
    max_out_of_order: timedelta = timedelta(minutes=10),
    late_sink: Callable[[Dict[str, Any]], None] | None = None,
) -> None:
    """Stream edges with bounded out-of-orderness and late-event replay.

    This helper mimics Flink's
    :meth:`WatermarkStrategy.forBoundedOutOfOrderness` with a
    ``max_out_of_order`` delay of 10 minutes by default.  Events that arrive
    later than the watermark (``max_ts - max_out_of_order``) are considered
    *late* and are forwarded to ``late_sink`` instead of being written to the
    hot layer.  On-time events are grouped into 30 second tumbling windows and
    flushed to ``hot_layer`` similarly to :func:`process_edge_stream`.

    Parameters
    ----------
    events:
        Iterable of event dictionaries with ``src``, ``dst`` and ``ts``
        (:class:`datetime`).  The iterable is consumed in the given order to
        simulate streaming arrival.
    hot_layer:
        Instance of :class:`RedisGraphHotLayer` receiving edge updates.
    window:
        Length of the tumbling window.  Defaults to 30 seconds.
    max_out_of_order:
        Maximum tolerated difference between the highest observed timestamp and
        the current event before the event is considered late.  Defaults to
        10 minutes.
    late_sink:
        Optional callback invoked with late events.  It can forward the event to
        a Kafka topic such as ``late_edges``.
    """

    max_ts: datetime | None = None
    batch: List[Dict[str, Any]] = []
    window_start: datetime | None = None

    for event in events:
        ts = event["ts"]
        if max_ts is None or ts > max_ts:
            max_ts = ts
        watermark = max_ts - max_out_of_order

        if ts < watermark:
            if LATE_EDGE_TOTAL is not None:  # pragma: no branch - metric optional
                LATE_EDGE_TOTAL.inc()
            if late_sink is not None:
                late_sink(event)
            continue

        if window_start is None:
            window_start = ts
        if ts - window_start >= window:
            _flush_batch(batch, hot_layer)
            batch = []
            window_start = ts

        batch.append(event)

    if batch:
        _flush_batch(batch, hot_layer)


def _flush_batch(batch: List[Dict[str, Any]], hot_layer: "RedisGraphHotLayer") -> None:
    """Write a batch of events to the hot layer."""
    for event in batch:
        hot_layer.add_edge(event["src"], event["dst"])


@dataclass
class RedisGraphHotLayer:
    """Simplistic in-memory stand-in for a RedisGraph hot layer.

    The class records the latency of each write/read operation to enable
    monitoring of the p95 latency.
    """

    edges: Dict[str, set] = None
    latencies: List[float] = None  # seconds

    def __post_init__(self) -> None:  # pragma: no cover - simple initialisation
        self.edges = defaultdict(set)
        self.latencies = []

    def add_edge(self, src: str, dst: str) -> None:
        """Insert an edge ``src -> dst``.

        The operation time is recorded to allow p95 latency computation.
        """
        start = time.perf_counter()
        self.edges[src].add(dst)
        self.latencies.append(time.perf_counter() - start)

    def neighbours(self, src: str) -> List[str]:
        """Return neighbours of ``src`` and record query latency."""
        start = time.perf_counter()
        result = list(self.edges.get(src, []))
        self.latencies.append(time.perf_counter() - start)
        return result

    def p95_latency_ms(self) -> float:
        """Compute the 95th percentile latency in milliseconds."""
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(0.95 * (len(sorted_lat) - 1))
        return sorted_lat[idx] * 1000


def laplacian(B: np.ndarray, W: np.ndarray | None = None) -> np.ndarray:
    """Compute the normalised Laplacian for a hypergraph edge type.

    Parameters
    ----------
    B:
        Incidence matrix with shape ``(n_nodes, n_edges)`` where ``B[i, j] = 1``
        if node ``i`` participates in hyperedge ``j``.  The matrix corresponds
        to :math:`B_t` in the specification.
    W:
        Diagonal matrix of hyperedge weights with shape ``(n_edges, n_edges)``.
        When ``None`` the identity matrix is used, matching the default
        :math:`W_t = I` (initial weight ``1`` for every hyperedge).

    Returns
    -------
    numpy.ndarray
        The Laplacian :math:`\mathcal{L}^{(t)}` defined as

        .. math::

           \mathcal{L}^{(t)} = I - D_t^{-1/2} B_t W_t D_t^{-1} B_t^\top D_t^{-1/2}

        where :math:`D_t` is the diagonal matrix of node degrees.
    """

    n_nodes, n_edges = B.shape
    if W is None:
        W = np.eye(n_edges)
    deg = B @ W @ np.ones(n_edges)
    D = np.diag(np.maximum(deg, 1e-12))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    D_inv = np.linalg.inv(D)
    return np.eye(n_nodes) - D_inv_sqrt @ B @ W @ D_inv @ B.T @ D_inv_sqrt


def multiplex_laplacian(
    B_list: List[np.ndarray],
    alpha: np.ndarray,
    W_list: List[np.ndarray] | None = None,
) -> np.ndarray:
    """Combine per-type Laplacians into a multiplex Laplacian.

    Parameters
    ----------
    B_list:
        List of incidence matrices for each edge type.
    alpha:
        Non‑negative weights for each type.  They are renormalised to ensure
        :math:`\sum_t \alpha_t = 1` before combining the Laplacians.
    W_list:
        Optional list of weight matrices corresponding to ``B_list``.  When
        ``None`` each edge type is assumed to have unit weights.

    Returns
    -------
    numpy.ndarray
        Multiplex Laplacian :math:`\Delta_\text{multi}` as in the checklist

        .. math::

           \Delta_\text{multi} = \sum_t \alpha_t\,\mathcal{L}^{(t)}.
    """

    if W_list is None:
        W_list = [None] * len(B_list)

    alpha = np.asarray(alpha, dtype=float)
    if np.any(alpha < 0):
        raise ValueError("alpha_t must be non-negative")
    if not np.isclose(alpha.sum(), 1.0):
        alpha = alpha / alpha.sum()

    Ls = [laplacian(B, W) for B, W in zip(B_list, W_list)]
    return sum(a * L for a, L in zip(alpha, Ls))
