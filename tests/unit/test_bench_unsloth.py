"""Ensure Unsloth benchmarks meet VRAM and speed targets."""

import json
from pathlib import Path

import pytest

BENCH_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "bench_unsloth.json"


@pytest.mark.parametrize("max_ratio", [0.6])
def test_vram_below_baseline(max_ratio: float) -> None:
    """VRAM used by Unsloth should be under the configured ratio."""
    data = json.loads(BENCH_PATH.read_text())
    ratio = data["unsloth_vram_mb"] / data["hf_vram_mb"]
    assert ratio < max_ratio, f"VRAM ratio {ratio:.2f} exceeds {max_ratio:.2f}"


def test_speedup_meets_target() -> None:
    """Benchmark should demonstrate at least 1.7x speedup."""
    data = json.loads(BENCH_PATH.read_text())
    assert data["speedup_vs_hf"] >= 1.7
