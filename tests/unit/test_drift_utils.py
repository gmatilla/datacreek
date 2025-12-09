"""Tests for simple embedding drift helper utilities."""

import datacreek.drift as drift


def test_drift_level_classification() -> None:
    """Values should map to the correct qualitative level."""

    assert drift.drift_level(0.05) == "ok"
    assert drift.drift_level(0.08) == "warn"
    assert drift.drift_level(0.12) == "crit"


def test_should_trigger_retrain() -> None:
    """Retraining is only triggered for critical drift values."""

    assert not drift.should_trigger_retrain(0.08)
    assert drift.should_trigger_retrain(0.11)
