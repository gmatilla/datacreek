"""Tests for the embedding CPU usage Prometheus metric and cost tracking."""

import time

from prometheus_client import REGISTRY

from datacreek.analysis.embedding import (
    embedding_cpu_seconds_total,
    track_embedding_cpu_seconds,
)
from metrics_prometheus.cpu_billing import CpuCostTracker, cpu_cost_total


def reset_metrics() -> None:
    embedding_cpu_seconds_total._metrics.clear()  # type: ignore[attr-defined]
    cpu_cost_total._metrics.clear()  # type: ignore[attr-defined]


def test_track_embedding_cpu_seconds_increments_counter() -> None:
    """Counter should increase by the elapsed time inside the context."""
    reset_metrics()
    with track_embedding_cpu_seconds("alice"):
        sum(i * i for i in range(1000))
    value = REGISTRY.get_sample_value(
        "embedding_cpu_seconds_total", {"tenant": "alice"}
    )
    assert value is not None and value > 0


def test_context_manager_updates_cost_when_tracker_provided(monkeypatch) -> None:
    """Supplying a cost tracker should increment the cost counter proportionally."""
    reset_metrics()

    times = iter([1.0, 3.0])  # 2 seconds elapsed
    monkeypatch.setattr(time, "perf_counter", lambda: next(times))
    tracker = CpuCostTracker(prices={"alice": 0.5})
    with track_embedding_cpu_seconds("alice", tracker):
        pass
    seconds_value = REGISTRY.get_sample_value(
        "embedding_cpu_seconds_total", {"tenant": "alice"}
    )
    cost_value = REGISTRY.get_sample_value("cpu_cost_total", {"tenant": "alice"})
    assert seconds_value == 2.0
    assert cost_value == 1.0
