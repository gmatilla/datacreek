"""Utilities for git-based rollback and SLA tracking."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path


def rollback_gremlin_diff(repo: str, output: str = "rollback.diff") -> str:
    """Return path to diff patch for the last commit.

    Parameters
    ----------
    repo:
        Path to the git repository to diff.
    output:
        Path where the diff patch will be written.
    """
    repo_path = Path(repo)
    patch = subprocess.check_output(
        ["git", "diff", "HEAD~1", "HEAD"], cwd=repo_path
    ).decode()
    out_path = repo_path / output
    out_path.write_text(patch)
    return str(out_path)


class SheafSLA:
    """Track mean time to recovery for the sheaf checker."""

    def __init__(self, threshold_hours: float = 2.0) -> None:
        self.threshold = threshold_hours * 3600.0
        self._fail_times: list[float] = []

    def record_failure(self, timestamp: float | None = None) -> None:
        self._fail_times.append(timestamp or time.time())

    def mttr_hours(self) -> float:
        if len(self._fail_times) < 2:
            return 0.0
        import numpy as np

        intervals = np.diff(self._fail_times)
        return float(np.mean(intervals)) / 3600.0

    def sla_met(self) -> bool:
        return self.mttr_hours() <= self.threshold / 3600.0
