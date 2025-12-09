import pytest

from datacreek.security.dp_budget import DPBudget, DPBudgetManager


def test_budget_consume_prune_remaining():
    b = DPBudget(epsilon=3.0, window=10.0)
    assert b.consume(1.0, now=0.0)
    # exceeding
    assert not b.consume(2.5, now=0.0)
    assert b.remaining(now=1.0) == pytest.approx(2.0)
    # after window, previous event pruned
    assert b.consume(2.5, now=11.0)
    assert b.remaining(now=11.0) == pytest.approx(0.5)


def test_budget_manager_lifecycle():
    mgr = DPBudgetManager(window_seconds=5.0)
    mgr.add_user("alice", 2.0)
    assert mgr.consume("alice", 1.0, now=0.0)
    assert not mgr.consume("alice", 1.5, now=1.0)
    assert mgr.remaining("alice", now=1.0) == pytest.approx(1.0)
    mgr.reset()
    assert mgr.remaining("alice", now=1.0) == pytest.approx(2.0)
    with pytest.raises(KeyError):
        mgr.consume("bob", 1.0)
