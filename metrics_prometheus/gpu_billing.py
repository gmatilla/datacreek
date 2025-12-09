"""GPU credit tracking, cost computation and quota enforcement.

This module exposes a :class:`QuotaController` that decrements per-tenant GPU
credit balances.  Consumed minutes are exported via the Prometheus counter
``gpu_minutes_total{tenant}`` and the corresponding cost is tracked through the
``gpu_cost_total{tenant}`` counter using the pricing formula

.. math::

    C = t_{GPU}(min) \times P_{unit}

where ``t_GPU`` is the amount of GPU minutes consumed and ``P_unit`` is the
per-minute price.  When a tenant exhausts their credit a
:class:`QuotaExceededError` is raised so callers can fail the job with a
``failed_quota`` state.
"""

from __future__ import annotations

from typing import Dict

from prometheus_client import Counter

# Counter recording the total GPU minutes consumed per tenant
# Labels: tenant - unique tenant identifier
# Metric name intentionally follows the ``*_total`` convention so Prometheus
# exposes ``gpu_minutes_total_total`` as the sample value.
gpu_minutes_total = Counter(
    "gpu_minutes_total", "Accumulated GPU minutes per tenant", ["tenant"]
)

# Counter recording the total GPU cost per tenant in the configured currency
gpu_cost_total = Counter(
    "gpu_cost_total", "Accumulated GPU cost per tenant", ["tenant"]
)


class QuotaExceededError(RuntimeError):
    """Raised when a tenant attempts to consume more GPU minutes than available."""


class QuotaController:
    """Simple in-memory quota tracker for GPU credits and cost.

    Parameters
    ----------
    credits:
        Mapping of tenant identifier to remaining GPU minutes.
    prices:
        Mapping of tenant identifier to GPU price per minute.  Tenants not
        present default to a price of ``0`` which effectively disables cost
        tracking for them.
    """

    def __init__(
        self, credits: Dict[str, float], prices: Dict[str, float] | None = None
    ) -> None:
        # Store mutable copies so callers can pass in literals without side effects.
        self._credits: Dict[str, float] = dict(credits)
        self._prices: Dict[str, float] = dict(prices or {})

    def get_remaining(self, tenant: str) -> float:
        """Return the remaining GPU minutes for *tenant*.

        Tenants with no recorded balance default to ``0`` which allows callers
        to query a tenant prior to allocating any credits.
        """

        return self._credits.get(tenant, 0.0)

    def add_credits(self, tenant: str, minutes: float) -> float:
        """Add ``minutes`` of GPU credit for *tenant* and return new balance."""

        new_balance = self._credits.get(tenant, 0.0) + minutes
        self._credits[tenant] = new_balance
        return new_balance

    def set_price(self, tenant: str, price: float) -> float:
        """Set GPU price per minute for *tenant* and return the new price.

        Updating the price allows subsequent consumption to reflect revised
        billing rates, enabling administrators to adjust tenant-specific
        charges without recreating the controller.
        """

        self._prices[tenant] = price
        return price

    def consume(self, tenant: str, minutes: float) -> float:
        """Consume GPU minutes for *tenant* and update Prometheus metrics.

        Parameters
        ----------
        tenant:
            Tenant identifier whose credit balance should be decremented.
        minutes:
            Number of minutes of GPU time being consumed.

        Returns
        -------
        float
            Remaining credits after consumption.

        Raises
        ------
        QuotaExceededError
            If the tenant lacks sufficient credits to cover ``minutes``.
        """

        remaining = self._credits.get(tenant, 0.0)
        if remaining < minutes:
            # Do not modify metrics or balance; signal to caller that the job
            # must fail with a quota error.
            raise QuotaExceededError(f"tenant {tenant} exceeded available GPU credits")

        remaining -= minutes
        self._credits[tenant] = remaining

        # Record usage in Prometheus for billing and monitoring purposes.
        gpu_minutes_total.labels(tenant=tenant).inc(minutes)

        # Compute incurred cost using the configured per-minute price
        # and record it in Prometheus.  This enables downstream billing
        # systems to scrape a monetary total directly.
        price = self._prices.get(tenant, 0.0)
        gpu_cost_total.labels(tenant=tenant).inc(minutes * price)
        return remaining


def calculate_cost(minutes: float, price_per_minute: float) -> float:
    """Return the monetary cost for ``minutes`` of GPU time.

    This helper simply applies :math:`C = t_{GPU} \times P_{unit}` and is used
    in tests and external billing utilities.
    """

    return minutes * price_per_minute
