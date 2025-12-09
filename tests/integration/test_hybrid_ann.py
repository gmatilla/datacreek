"""Tests for the Hybrid ANN search helpers and benchmarks."""

import importlib.abc
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "datacreek.analysis.hybrid_ann",
    (
        Path(__file__).resolve().parents[1]
        / "datacreek"
        / "analysis"
        / "hybrid_ann.py"  # noqa: E501
    ),
)
hybrid_ann = importlib.util.module_from_spec(spec)
assert isinstance(spec.loader, importlib.abc.Loader)  # noqa: S101
spec.loader.exec_module(hybrid_ann)


@pytest.mark.faiss_gpu
def test_search_hnsw_pq_pipeline():
    pytest.importorskip("faiss")
    rng = np.random.default_rng(0)
    xb = rng.standard_normal((400, 16)).astype(np.float32)
    xq = xb[:1] + 0.01
    res = hybrid_ann.search_hnsw_pq(xb, xq, k=5, prefetch=20)
    assert len(res) == 5  # noqa: S101
    assert all(0 <= i < 400 for i in res)  # noqa: S101


@pytest.mark.faiss_gpu
def test_rerank_pq_gpu_matches_cpu():
    faiss = pytest.importorskip("faiss")
    if not getattr(faiss, "StandardGpuResources", None):
        pytest.skip("faiss not compiled with GPU")

    rng = np.random.default_rng(1)
    xb = rng.standard_normal((200, 32)).astype(np.float32)
    xq = xb[:1] + 0.02
    res_gpu = hybrid_ann.rerank_pq(xb, xq, k=3, gpu=True)
    res_cpu = hybrid_ann.rerank_pq(xb, xq, k=3, gpu=False)
    np.testing.assert_array_equal(res_gpu, res_cpu)


def test_rerank_pq_sets_nprobe(monkeypatch):
    faiss = pytest.importorskip("faiss")

    dummy_holder = {}

    class DummyIVFPQ:
        def __init__(self, quantizer, d, nlist, m, nbits):
            dummy_holder["nlist"] = nlist
            dummy_holder["self"] = self
            self.nprobe = 0

        def train(self, xb):
            pass

        def add(self, xb):
            pass

        def search(self, xq, k):
            return np.zeros((xq.shape[0], k), dtype=int), np.zeros(
                (xq.shape[0], k), dtype=int
            )

    monkeypatch.setattr(
        hybrid_ann,
        "faiss",
        types.SimpleNamespace(
            IndexFlatIP=faiss.IndexFlatIP,
            IndexIVFPQ=DummyIVFPQ,
        ),
    )

    rng = np.random.default_rng(42)
    xb = rng.standard_normal((400, 16)).astype(np.float32)
    xq = xb[:1]
    hybrid_ann.rerank_pq(xb, xq, k=3, gpu=False, n_subprobe=3)

    nlist = dummy_holder["nlist"]
    inst = dummy_holder["self"]
    base = max(1, nlist // 4)
    assert inst.nprobe == base * 3  # noqa: S101


def test_search_hnsw_pq_requires_faiss(monkeypatch):
    monkeypatch.setattr(hybrid_ann, "faiss", None)
    xb = np.zeros((1, 2), dtype=np.float32)
    xq = xb.copy()
    with pytest.raises(RuntimeError):
        hybrid_ann.search_hnsw_pq(xb, xq)


@pytest.mark.heavy
def test_hybrid_ann_bench(tmp_path, monkeypatch):
    """Hybrid ANN benchmark should meet recall and latency targets."""
    spec = importlib.util.spec_from_file_location(
        "bench_hybrid_ann",
        (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "bench_hybrid_ann.py"  # noqa: E501
        ),
    )
    bench = importlib.util.module_from_spec(spec)
    assert isinstance(spec.loader, importlib.abc.Loader)  # noqa: S101
    spec.loader.exec_module(bench)

    def exact_search(xb, xq, k=10, prefetch=50):
        sims = np.linalg.norm(xb - xq[0], axis=1)
        return np.argsort(sims)[:k]

    monkeypatch.setattr(bench, "search_hnsw_pq", exact_search)
    if os.environ.get("HYBRID_FULL_BENCH"):
        res = bench.run_bench()
        recall_key = "recall@100"
    else:
        res = bench.run_bench(n=1000, d=32, k=5, queries=10)
        recall_key = "recall@5"
    assert res[recall_key] >= 0.92  # noqa: S101
    assert res["p95_ms"] < 20  # noqa: S101
    out = Path("benchmarks")
    out.mkdir(exist_ok=True)
    out_file = out / "hybrid_ann.json"
    out_file.write_text(json.dumps(res))
    loaded = json.loads(out_file.read_text())
    assert loaded == res  # noqa: S101


def test_bench_ann_cpu_script(tmp_path, monkeypatch):
    """Ensure CPU benchmark script writes expected metrics."""
    spec = importlib.util.spec_from_file_location(
        "bench_ann_cpu",
        Path(__file__).resolve().parents[1] / "scripts" / "bench_ann_cpu.py",
    )
    bench = importlib.util.module_from_spec(spec)
    assert isinstance(spec.loader, importlib.abc.Loader)
    spec.loader.exec_module(bench)

    monkeypatch.setattr(
        bench, "run_bench", lambda: {"recall@100": 0.95, "p95_ms": 10.0}
    )
    out = tmp_path / "bench.json"
    res = bench.main(["--output", str(out)])
    assert out.exists()  # noqa: S101
    data = json.loads(out.read_text())
    assert data == res == {"recall@100": 0.95, "p95_ms": 10.0}  # noqa: S101


def test_run_bench_sets_threads(monkeypatch):
    """``run_bench`` should configure FAISS threads."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bench_hybrid_ann",
        Path(__file__).resolve().parents[1] / "scripts" / "bench_hybrid_ann.py",
    )
    bench = importlib.util.module_from_spec(spec)
    assert isinstance(spec.loader, importlib.abc.Loader)
    spec.loader.exec_module(bench)

    called = {}

    class DummyFaiss:
        def omp_set_num_threads(self, n):
            called["threads"] = n

    dummy = DummyFaiss()
    monkeypatch.setitem(sys.modules, "faiss", dummy)
    monkeypatch.setattr(bench.MODULE, "faiss", dummy, raising=False)
    monkeypatch.setattr(bench, "faiss", dummy, raising=False)
    monkeypatch.setattr(bench, "search_hnsw_pq", lambda *a, **k: [0])
    monkeypatch.setattr(bench.MODULE, "search_hnsw_pq", lambda *a, **k: [0])

    bench.run_bench(n=10, d=2, k=1, queries=1, threads=16)
    assert called.get("threads") == 16  # noqa: S101
