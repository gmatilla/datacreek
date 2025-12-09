"""Validate heavy dataset benchmark durations are recorded."""

import json
from pathlib import Path

BENCH_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "bench_datasets.json"


def test_benchmarks_have_positive_durations() -> None:
    """Recorded epoch durations should be positive numbers."""
    data = json.loads(BENCH_PATH.read_text())
    assert data["tinystories_epoch_s"] > 0
    assert data["dbpedia_epoch_s"] > 0
