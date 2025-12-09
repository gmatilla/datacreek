#!/usr/bin/env python3
"""Benchmark Hybrid ANN on CPU and record metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.bench_hybrid_ann import run_bench


def main(argv: list[str] | None = None) -> dict[str, float]:
    """Run benchmark and write results to file."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/bench_ann_cpu.json")
    )
    args = parser.parse_args(argv)
    result = run_bench()
    args.output.parent.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(result))
    print(json.dumps(result))
    return result


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
