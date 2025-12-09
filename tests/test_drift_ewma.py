"""Tests for adaptive drift threshold based on EWMA per tenant."""

from math import sqrt

import pytest
from prometheus_client import REGISTRY

from datacreek.drift import TenantEWMA, drift_alert_total


def test_ewma_updates_state_correctly() -> None:
    det = TenantEWMA(lambda_=0.2)
    # First observation initializes the state without alert
    assert not det.update("alice", 0.0)
    # Second observation updates mean and sigma
    alerted = det.update("alice", 0.5)
    assert not alerted
    state = det._state["alice"]
    expected_mu = 0.2 * 0.5 + 0.8 * 0.0
    expected_sigma = sqrt(0.2 * (0.5 - expected_mu) ** 2 + 0.8 * 0.0**2)
    assert state.mu == pytest.approx(expected_mu)
    assert state.sigma == pytest.approx(expected_sigma)


def test_alert_triggers_on_large_drift() -> None:
    det = TenantEWMA(lambda_=0.1)
    drift_alert_total.clear()
    for _ in range(5):
        assert not det.update("bob", 0.1)
    assert det.update("bob", 1.0)
    sample = REGISTRY.get_sample_value("drift_alert_total", labels={"tenant": "bob"})
    assert sample == 1
