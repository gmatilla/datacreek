import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datacreek.security.dp_budget import DPBudgetManager


def test_dp_budget_manager():
    mgr = DPBudgetManager()
    mgr.add_user("alice", 1.0)
    assert mgr.remaining("alice") == 1.0
    ok = mgr.consume("alice", 0.3)
    assert ok is True
    assert mgr.remaining("alice") == pytest.approx(0.7)
    ok = mgr.consume("alice", 0.8)
    assert ok is False
    mgr.reset()
    assert mgr.remaining("alice") == 1.0


def test_dp_budget_sliding(monkeypatch):
    mgr = DPBudgetManager(window_seconds=10.0)
    mgr.add_user("bob", 1.0)

    monkeypatch.setattr("time.time", lambda: 0.0)
    assert mgr.consume("bob", 0.6) is True
    monkeypatch.setattr("time.time", lambda: 5.0)
    assert mgr.consume("bob", 0.5) is False
    monkeypatch.setattr("time.time", lambda: 11.0)
    assert mgr.consume("bob", 0.5) is True
