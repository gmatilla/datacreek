import numpy as np
import pytest

from datacreek.analysis import nprobe_tuning


def _build_ivfpq(xb):
    faiss = pytest.importorskip("faiss")
    d = xb.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, 1, 1, 2)
    index.train(xb)
    index.add(xb)
    index.nprobe = 32
    return index


@pytest.mark.faiss_gpu
def test_autotune_nprobe_runs():
    faiss = pytest.importorskip("faiss")
    rng = np.random.default_rng(0)
    xb = rng.standard_normal((100, 8)).astype(np.float32)
    xq = xb[:5] + 0.01
    index = _build_ivfpq(xb)
    best = nprobe_tuning.autotune_nprobe(index, xb, xq, k=5, target=0.5, max_evals=5)
    assert 32 <= best <= 256
    assert index.nprobe == best


@pytest.mark.faiss_gpu
def test_autotune_nprobe_stops_on_target(monkeypatch):
    faiss = pytest.importorskip("faiss")
    xb = np.eye(4, dtype=np.float32)
    xq = xb.copy()
    index = _build_ivfpq(xb)

    class FakeOpt:
        def __init__(self):
            self.calls = 0

        def ask(self):
            self.calls += 1
            return [32]

        def tell(self, *args, **kwargs):
            pass

    fake_opt = FakeOpt()
    monkeypatch.setattr(nprobe_tuning, "Optimizer", lambda *a, **k: fake_opt)
    monkeypatch.setattr(nprobe_tuning, "_compute_recall", lambda *a, **k: 0.95)

    best = nprobe_tuning.autotune_nprobe(index, xb, xq, k=2, target=0.9, max_evals=5)
    assert best == 32
    assert fake_opt.calls == 1


@pytest.mark.faiss_gpu
def test_profile_nprobe(tmp_path):
    faiss = pytest.importorskip("faiss")
    rng = np.random.default_rng(1)
    xb = rng.standard_normal((50, 8)).astype(np.float32)
    xq = xb[:5] + 0.01
    index = _build_ivfpq(xb)
    out_path = tmp_path / "curve.pkl"
    res = nprobe_tuning.profile_nprobe(
        index, xb, xq, k=5, nprobes=[32, 64], path=out_path
    )
    assert list(res["nprobe"]) == [32, 64]
    assert len(res["latency"]) == 2
    assert out_path.exists()
