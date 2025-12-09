"""Differential privacy budget tracker.

This module provides a minimal sliding-window mechanism for
tracking how much privacy budget (epsilon) each user has consumed.
Events older than ``window_seconds`` no longer count against the
budget, effectively implementing a rolling allowance (e.g. per day).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Tuple


@dataclass
class DPBudget:
    """Per-user differential privacy budget with sliding window."""

    epsilon: float
    window: float = 86_400.0  # sliding window in seconds (default 24h)
    events: Deque[Tuple[float, float]] = field(default_factory=deque)

    def _prune(self, now: float) -> None:
        """Drop events outside the sliding window."""
        limit = now - self.window
        while self.events and self.events[0][0] <= limit:
            self.events.popleft()

    def _spent(self, now: float) -> float:
        self._prune(now)
        return sum(amount for _, amount in self.events)

    def consume(self, amount: float, *, now: float | None = None) -> bool:
        """Consume ``amount`` if budget allows within the window."""
        now = time.time() if now is None else now
        if self._spent(now) + amount > self.epsilon:
            return False
        self.events.append((now, amount))
        return True

    def remaining(self, *, now: float | None = None) -> float:
        """Return remaining epsilon within the window."""
        now = time.time() if now is None else now
        return self.epsilon - self._spent(now)


@dataclass
class DPBudgetManager:
    """Manage privacy budgets per user with sliding windows."""

    window_seconds: float = 86_400.0
    budgets: Dict[str, DPBudget] = field(default_factory=dict)

    def add_user(self, user: str, epsilon: float) -> None:
        """Register ``user`` with a sliding-window epsilon budget."""
        self.budgets[user] = DPBudget(epsilon, window=self.window_seconds)

    def consume(self, user: str, amount: float, *, now: float | None = None) -> bool:
        """Consume ``amount`` from ``user``'s budget."""
        if user not in self.budgets:
            raise KeyError(user)
        return self.budgets[user].consume(amount, now=now)

    def remaining(self, user: str, *, now: float | None = None) -> float:
        """Return remaining epsilon for ``user``."""
        if user not in self.budgets:
            raise KeyError(user)
        return self.budgets[user].remaining(now=now)

    def reset(self) -> None:
        """Drop all recorded events for every user."""
        for budget in self.budgets.values():
            budget.events.clear()
