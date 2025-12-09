"""CPU cost tracking for billing dashboards.

This module provides a :class:`CpuCostTracker` that records per-tenant CPU
usage and translates it to monetary cost.  The tracked data complements the
GPU billing counters to surface a unified CPU+GPU cost view.

The cost for a span of CPU time is computed with

.. math::
    C = t_{CPU}(s) \times P_{unit}

where ``t_CPU`` is measured in seconds and ``P_unit`` is the per-second price
configured for the tenant.  Recorded usage is exported via the Prometheus
counters :data:`cpu_seconds_total` and :data:`cpu_cost_total`.
"""

from __future__ import annotations

from typing import Dict

from prometheus_client import Counter

# Counter accumulating raw CPU seconds per tenant.  This metric follows the
# ``*_total`` naming convention so Prometheus exposes
# ``cpu_seconds_total_total`` as the sample name.
cpu_seconds_total = Counter(
    "cpu_seconds_total", "CPU seconds consumed per tenant", ["tenant"]
)

# Counter accumulating the monetary CPU cost per tenant in the configured
# currency.  Downstream billing dashboards can directly scrape this value to
# display total charges.
cpu_cost_total = Counter(
    "cpu_cost_total", "Accumulated CPU cost per tenant", ["tenant"]
)


class CpuCostTracker:
    """Track CPU usage and cost for a set of tenants.

    Parameters
    ----------
    prices:
        Optional mapping of tenant identifier to per-second CPU price.  Tenants
        not present default to a price of ``0`` which effectively disables cost
        tracking for them.
    """

    def __init__(self, prices: Dict[str, float] | None = None) -> None:
        # Copy mappings so callers can pass in literals without side effects.
        self._prices: Dict[str, float] = dict(prices or {})

    def set_price(self, tenant: str, price: float) -> float:
        """Set the per-second CPU price for *tenant* and return the new price."""

        self._prices[tenant] = price
        return price

    def record(self, tenant: str, seconds: float) -> None:
        """Record ``seconds`` of CPU time for *tenant* and update metrics."""

        cpu_seconds_total.labels(tenant=tenant).inc(seconds)
        price = self._prices.get(tenant, 0.0)
        cpu_cost_total.labels(tenant=tenant).inc(seconds * price)
