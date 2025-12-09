"""Utilities for detecting embedding drift and scheduling retraining.

This module exposes small helpers to compute a drift *level* given a
numeric score and decide whether retraining should be triggered.  The
thresholds are derived from the SaaS readiness checklist:

* warn when drift exceeds 0.07
* trigger critical alert and retraining when drift exceeds 0.10

The functions are intentionally tiny so they can be reused both in Airflow
DAGs and in unit tests without pulling heavy dependencies such as NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict

from prometheus_client import Counter

WARN_THRESHOLD: float = 0.07
"""Drift value above which the system should emit a warning."""

CRIT_THRESHOLD: float = 0.10
"""Drift value above which the system must trigger retraining."""

# Counter tracking how many drift alerts were raised for each tenant.  This can
# be scraped by billing or monitoring systems to observe stability across
# tenants over time.
drift_alert_total = Counter(
    "drift_alert_total", "Number of EWMA drift alerts by tenant", ["tenant"]
)


def drift_level(value: float) -> str:
    """Classify the drift *value* into qualitative buckets.

    Parameters
    ----------
    value:
        Measured drift magnitude, typically a distance between
        recent and reference embedding distributions.

    Returns
    -------
    str
        One of ``"ok"``, ``"warn"`` or ``"crit"`` according to the
        thresholds defined in this module.
    """

    if value >= CRIT_THRESHOLD:
        return "crit"
    if value >= WARN_THRESHOLD:
        return "warn"
    return "ok"


def should_trigger_retrain(value: float) -> bool:
    """Return ``True`` when the *value* requires starting retraining.

    This is a convenience wrapper around :func:`drift_level` that makes the
    intent explicit when used by orchestration code.
    """

    return value >= CRIT_THRESHOLD


@dataclass
class _EWMAState:
    """Internal container holding the running statistics for a tenant."""

    mu: float
    sigma: float


class TenantEWMA:
    """Track per-tenant drift with an Exponentially Weighted Moving Average.

    The detector maintains the mean :math:`\mu_t` and standard deviation
    :math:`\sigma_t` of observed drift scores for each tenant using the
    recursive formulas:

    .. math::

        \mu_t = \lambda d_t + (1 - \lambda) \mu_{t-1}\\
        \sigma_t = \sqrt{\lambda (d_t - \mu_t)^2 + (1-\lambda) \sigma_{t-1}^2}

    An alert is raised when the new drift score ``d_t`` exceeds the dynamic
    threshold ``mu_t + 3 * sigma_t``.  This scheme adapts to each tenant's
    typical drift level and reduces false positives for low-traffic tenants.

    Parameters
    ----------
    lambda_:
        Smoothing factor :math:`\lambda` in ``[0, 1]`` controlling how much
        weight is given to the latest observation.
    """

    def __init__(self, lambda_: float) -> None:
        self.lambda_ = lambda_
        self._state: Dict[str, _EWMAState] = {}

    def update(self, tenant: str, drift: float) -> bool:
        """Update the running statistics for ``tenant`` with ``drift``.

        Parameters
        ----------
        tenant:
            Identifier for the tenant emitting the drift score.
        drift:
            Current drift measurement ``d_t``.

        Returns
        -------
        bool
            ``True`` if the updated score exceeds ``mu_t + 3*sigma_t`` and an
            alert should be triggered, ``False`` otherwise.
        """

        state = self._state.get(tenant)
        if state is None:
            # First observation initialises the state without alerting.
            self._state[tenant] = _EWMAState(mu=drift, sigma=0.0)
            return False

        mu_t = self.lambda_ * drift + (1 - self.lambda_) * state.mu
        sigma_t = sqrt(
            self.lambda_ * (drift - mu_t) ** 2 + (1 - self.lambda_) * state.sigma**2
        )
        alert = drift > mu_t + 3 * sigma_t
        self._state[tenant] = _EWMAState(mu=mu_t, sigma=sigma_t)
        if alert:
            drift_alert_total.labels(tenant=tenant).inc()
        return alert


__all__ = [
    "WARN_THRESHOLD",
    "CRIT_THRESHOLD",
    "drift_level",
    "should_trigger_retrain",
    "drift_alert_total",
    "TenantEWMA",
]
