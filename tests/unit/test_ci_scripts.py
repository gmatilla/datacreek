"""Unit tests for CI helper scripts."""

from pathlib import Path


def test_run_changed_tests_filters_unit():
    """run_changed_tests.sh should only consider unit tests."""
    content = Path("scripts/run_changed_tests.sh").read_text()
    assert "tests/unit" in content
