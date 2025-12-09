"""Tests for CPU cost Prometheus exporter and cost tracker."""

from prometheus_client import REGISTRY

from metrics_prometheus.cpu_billing import (
    CpuCostTracker,
    cpu_cost_total,
    cpu_seconds_total,
)


def reset_metrics() -> None:
    """Utility to clear counter state between tests."""
    cpu_seconds_total._metrics.clear()  # type: ignore[attr-defined]
    cpu_cost_total._metrics.clear()  # type: ignore[attr-defined]


def test_record_increments_counters_and_cost() -> None:
    reset_metrics()
    tracker = CpuCostTracker(prices={"acme": 0.02})
    tracker.record("acme", 5.0)
    seconds_value = REGISTRY.get_sample_value("cpu_seconds_total", {"tenant": "acme"})
    cost_value = REGISTRY.get_sample_value("cpu_cost_total", {"tenant": "acme"})
    assert seconds_value == 5.0
    assert cost_value == 0.1


def test_set_price_controls_subsequent_cost() -> None:
    reset_metrics()
    tracker = CpuCostTracker(prices={"acme": 0.01})
    tracker.set_price("acme", 0.05)
    tracker.record("acme", 2.0)
    cost_value = REGISTRY.get_sample_value("cpu_cost_total", {"tenant": "acme"})
    assert cost_value == 0.1
