#!/usr/bin/env python3
"""Aggregate benchmark results and system metrics."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCH_DIR = pathlib.Path(os.environ.get("BENCHMARK_DIR", ROOT / "benchmarks"))
if str(ROOT / "scripts") in sys.path:
    sys.path.remove(str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import argparse
import importlib
import importlib.util
import json
import time

psutil_spec = importlib.util.find_spec("psutil")
if psutil_spec:
    import psutil  # type: ignore
else:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

np = None
if importlib.util.find_spec("numpy") is not None:
    import numpy as np  # type: ignore


def check_regression(
    metrics: dict[str, float], baseline: dict[str, float], threshold: float = 0.05
) -> None:
    """Raise SystemExit if metrics regress beyond threshold.

    Latency, CPU and memory must not exceed baseline by more than
    ``threshold``. Recall must not drop below the baseline minus
    ``threshold``.
    """
    regressions = []
    if metrics["p95_latency"] > baseline["p95_latency"] * (1 + threshold):
        regressions.append("p95_latency")
    if metrics["cpu_percent"] > baseline["cpu_percent"] * (1 + threshold):
        regressions.append("cpu_percent")
    if metrics["memory_mb"] > baseline["memory_mb"] * (1 + threshold):
        regressions.append("memory_mb")
    if metrics["recall"] < baseline["recall"] * (1 - threshold):
        regressions.append("recall")
    for key in ("ingest_rate", "graphwave_rate"):
        if (
            key in metrics
            and key in baseline
            and metrics[key] < baseline[key] * (1 - threshold)
        ):
            regressions.append(key)
    if (
        "whisper_xrt" in metrics
        and "whisper_xrt" in baseline
        and metrics["whisper_xrt"] > baseline["whisper_xrt"] * (1 + threshold)
    ):
        regressions.append("whisper_xrt")
    if (
        "pid_time" in metrics
        and "pid_time" in baseline
        and metrics["pid_time"] > baseline["pid_time"] * (1 + threshold)
    ):
        regressions.append("pid_time")
    if regressions:
        reg = ", ".join(regressions)
        raise SystemExit(f"Benchmark regression detected in: {reg}")


def benchmark(
    backend: str = "flat", n: int = 1000, d: int = 128, k: int = 5
) -> tuple[float, float]:
    """Return (p95 latency, recall@k) for a random graph."""
    if np is None:
        raise SystemExit("bench_all requires numpy")
    from datacreek.core.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph()
    for i in range(n):
        kg.graph.add_node(f"n{i}", embedding=np.random.rand(d).tolist())

    kg.build_faiss_index(method=backend)

    q = np.random.rand(d)
    latencies: list[float] = []
    recall_hits = 0
    vectors = np.vstack([kg.graph.nodes[n]["embedding"] for n in kg.graph.nodes])
    base = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q))
    true_top = set(np.argsort(base)[-k:])

    for _ in range(100):
        start = time.monotonic()
        res = kg.search_faiss(q, k=k)
        latencies.append(time.monotonic() - start)
        idxs = [int(r[1:]) for r in res]
        recall_hits += len(true_top.intersection(idxs))

    p95 = float(np.quantile(latencies, 0.95))
    recall = recall_hits / (100 * k)
    return p95, recall


def ingest_rate(iters: int = 1000) -> float:
    """Measure how many ingest queue operations per second are sustained."""
    from datacreek.utils import backpressure

    backpressure.set_limit(iters)
    start = time.monotonic()
    for _ in range(iters):
        backpressure.acquire_slot()
        backpressure.release_slot()
    return iters / (time.monotonic() - start)


def graphwave_rate(n: int = 256) -> float:
    """Measure GraphWave embedding throughput in nodes per second."""
    if np is None:
        raise SystemExit("bench_all requires numpy")
    import networkx as nx

    from datacreek.analysis.fractal import graphwave_embedding_chebyshev

    g = nx.path_graph(n)
    start = time.monotonic()
    graphwave_embedding_chebyshev(g, scales=[0.5], num_points=4, order=3)
    return n / (time.monotonic() - start)


def whisper_xrt_metric() -> float:
    """Return realtime factor for a minimal Whisper batch."""
    if np is None:
        raise SystemExit("bench_all requires numpy")
    from datacreek.utils.whisper_batch import transcribe_audio_batch

    audio = np.zeros(16000, dtype=np.float32)
    start = time.monotonic()
    transcribe_audio_batch([(audio, 16000)])
    return (time.monotonic() - start) / 1.0


def pid_convergence_time() -> float:
    """Simulate TTL PID convergence time."""
    target = 0.8
    kp, ki, i_max = 500.0, 0.05, 3600.0
    ttl = 3600
    integral = 0.0
    ema = 0.5
    alpha = 0.3
    ratio = 0.5
    start = time.monotonic()
    for _ in range(50):
        ema = alpha * ratio + (1 - alpha) * ema
        error = ema - target
        integral = max(-i_max, min(i_max, integral + error * 300))
        delta = kp * error + ki * integral
        ttl = max(300, min(7200, int(ttl + delta)))
        ratio += (target - ratio) * 0.2
        if abs(error) <= 0.05:
            break
    return time.monotonic() - start


def gpu_count() -> int:
    """Return available GPU count if FAISS supports it."""
    try:
        import faiss

        return faiss.get_num_gpus()
    except Exception:  # pragma: no cover - optional dependency
        return 0


def build_trend(path: pathlib.Path) -> None:
    """Write a markdown table summarizing benchmark history."""
    files = sorted(BENCH_DIR.glob("bench_*.json"))
    if not files:
        return
    lines = [
        "| commit | ingest_rate | graphwave_rate | whisper_xrt | pid_time |",
        "| ------ | -----------:| -------------:| ----------:| --------:|",
    ]
    for f in files:
        data = json.loads(f.read_text())
        sha = f.stem.split("_", 1)[1]
        lines.append(
            f"| {sha} | {data.get('ingest_rate',0):.2f} | {data.get('graphwave_rate',0):.2f} | {data.get('whisper_xrt',0):.2f} | {data.get('pid_time',0):.2f} |"
        )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="flat")
    parser.add_argument("--baseline", type=pathlib.Path)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--trend", action="store_true")
    args = parser.parse_args()
    if args.trend:
        out = args.output or BENCH_DIR / "bench_trend.md"
        BENCH_DIR.mkdir(exist_ok=True)
        build_trend(out)
        return
    if psutil is None:
        raise SystemExit("bench_all requires psutil")
    proc = psutil.Process()
    cpu_before = proc.cpu_percent(interval=None)
    mem_before = proc.memory_info().rss / 1e6
    p95, recall = benchmark(args.backend)
    cpu_after = proc.cpu_percent(interval=None)
    mem_after = proc.memory_info().rss / 1e6
    metrics = {
        "cpu_percent": cpu_after - cpu_before,
        "memory_mb": mem_after - mem_before,
        "gpu_count": gpu_count(),
        "p95_latency": p95,
        "recall": recall,
        "ingest_rate": ingest_rate(),
        "graphwave_rate": graphwave_rate(),
        "whisper_xrt": whisper_xrt_metric(),
        "pid_time": pid_convergence_time(),
    }
    result = json.dumps(metrics)
    print(result)
    if args.output:
        args.output.write_text(result)
    if args.record:
        sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .strip()
            .decode()
        )
        BENCH_DIR.mkdir(exist_ok=True)
        path = BENCH_DIR / f"bench_{sha}.json"
        path.write_text(result)
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        check_regression(metrics, baseline, args.threshold)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
