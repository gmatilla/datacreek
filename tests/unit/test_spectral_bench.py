"""Ensure spectral conv benchmark meets target improvements."""

import json
from pathlib import Path


def test_spectral_benchmark_targets():
    data = json.loads(Path("benchmarks/spectral_perf.json").read_text())
    assert (
        data["optimized_ms"] <= 0.8 * data["baseline_ms"]
    ), "runtime should drop by at least 20%"
    assert (
        data["macro_f1_new"] - data["macro_f1_base"] >= 0.005
    ), "macro-F1 should improve by ≥0.5 pt"
