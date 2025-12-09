"""Heavy tests for ANN CPU multi-probing."""

import json
import types

import numpy as np
import pytest

import datacreek.analysis.hybrid_ann as ha


@pytest.mark.heavy
def test_rerank_pq_sets_nprobe_multi(monkeypatch):
    """IVFPQ index should configure ``nprobe_multi`` for each subquantizer."""
    faiss = pytest.importorskip("faiss")

    holder: dict[str, object] = {}

    class DummyIVFPQ:
        def __init__(self, quantizer, d, nlist, m, nbits):
            holder["nlist"] = nlist
            holder["M"] = m
            self.pq = types.SimpleNamespace(M=m)
            self.nprobe = 0
            self.nprobe_multi = None
            holder["self"] = self

        def train(self, xb):
            pass

        def add(self, xb):
            pass

        def search(self, xq, k):
            return np.zeros((xq.shape[0], k), dtype=int), np.zeros((xq.shape[0], k))

    monkeypatch.setattr(
        ha,
        "faiss",
        types.SimpleNamespace(IndexFlatIP=faiss.IndexFlatIP, IndexIVFPQ=DummyIVFPQ),
    )

    rng = np.random.default_rng(0)
    xb = rng.standard_normal((400, 16)).astype(np.float32)
    xq = xb[:1]
    ha.rerank_pq(xb, xq, k=3, gpu=False)

    expected = ha.choose_nprobe_multi(holder["nlist"], holder["M"])
    assert holder["self"].nprobe_multi == [expected] * holder["M"]


@pytest.mark.heavy
def test_load_ivfpq_cpu_sets_nprobe_multi(monkeypatch, tmp_path):
    """``load_ivfpq_cpu`` should load and configure ``nprobe_multi``."""
    faiss = pytest.importorskip("faiss")

    class DummyIndex:
        def __init__(self):
            self.nsq = 3
            self.nprobe_multi = None

    monkeypatch.setattr(
        ha, "faiss", types.SimpleNamespace(read_index=lambda p: DummyIndex())
    )

    idx = ha.load_ivfpq_cpu(str(tmp_path / "x.faiss"), 7)
    assert idx.nprobe_multi == [7] * 3


@pytest.mark.heavy
def test_bench_ann_cpu_json(tmp_path):
    """``bench_ann_cpu`` should export recall and latency JSON."""
    pytest.importorskip("faiss")
    from scripts import bench_hybrid_ann

    out = tmp_path / "res.json"
    res = bench_hybrid_ann.run_bench(200, 16, 5, 5, threads=1)
    out.write_text(json.dumps(res))

    loaded = json.loads(out.read_text())
    assert set(loaded) == {"recall@5", "p95_ms"}
