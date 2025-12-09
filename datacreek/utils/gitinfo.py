"""Provide helpers to retrieve Git revision information."""

import subprocess
from pathlib import Path


def get_commit_hash() -> str:
    """Return the current git commit hash or ``"unknown"``."""
    try:
        root = Path(__file__).resolve().parents[1]
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"
