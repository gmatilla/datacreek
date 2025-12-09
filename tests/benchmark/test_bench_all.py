import importlib.util
import json
import os
import subprocess
import sys

import pytest

from scripts.bench_all import check_regression


def test_trend_generation(tmp_path):
    bench_dir = tmp_path / "benchmarks"
    bench_dir.mkdir()
    for i in range(3):
        (bench_dir / f"bench_sha{i}.json").write_text(
            json.dumps(
                {
                    "ingest_rate": 10 + i,
                    "graphwave_rate": 20 + i,
                    "whisper_xrt": 0.5 + i,
                    "pid_time": 0.1 * (i + 1),
                }
            )
        )

    import site

    env = {
        "PYTHONPATH": os.pathsep.join(site.getsitepackages() + ["."]),
        "BENCHMARK_DIR": str(bench_dir),
    }
    out_file = bench_dir / "trend.md"
    subprocess.check_call(
        [sys.executable, "scripts/bench_all.py", "--trend", "--output", str(out_file)],
        env=env,
    )
    assert out_file.exists()
    text = out_file.read_text()
    assert "sha0" in text


def test_bench_all_script(tmp_path):
    if any(
        importlib.util.find_spec(pkg) is None for pkg in ("pydantic", "numpy", "psutil")
    ):
        pytest.skip("missing heavy dependencies")
    import site

    env = {"PYTHONPATH": os.pathsep.join(site.getsitepackages() + ["."])}
    out = subprocess.check_output(
        [sys.executable, "scripts/bench_all.py", "--backend", "flat"],
        env=env,
    )
    metrics = json.loads(out)
    assert {
        "cpu_percent",
        "memory_mb",
        "gpu_count",
        "p95_latency",
        "recall",
        "ingest_rate",
        "graphwave_rate",
        "whisper_xrt",
        "pid_time",
    } <= metrics.keys()
    assert 0 <= metrics["recall"] <= 1
    assert metrics["p95_latency"] > 0


def test_check_regression():
    metrics = {
        "cpu_percent": 10,
        "memory_mb": 20,
        "gpu_count": 0,
        "p95_latency": 0.2,
        "recall": 0.9,
        "ingest_rate": 100.0,
        "graphwave_rate": 200.0,
        "whisper_xrt": 0.5,
        "pid_time": 0.2,
    }
    baseline = {
        "cpu_percent": 10,
        "memory_mb": 20,
        "gpu_count": 0,
        "p95_latency": 0.21,
        "recall": 0.88,
        "ingest_rate": 90.0,
        "graphwave_rate": 150.0,
        "whisper_xrt": 0.6,
        "pid_time": 0.3,
    }
    check_regression(metrics, baseline)
    baseline["recall"] = 0.95
    with pytest.raises(SystemExit):
        check_regression(metrics, baseline)
