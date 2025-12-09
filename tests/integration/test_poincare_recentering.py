try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None
import importlib.abc
import importlib.util
from pathlib import Path

import pytest

if np is not None:
    spec = importlib.util.spec_from_file_location(
        "poincare_recentering",
        Path(__file__).resolve().parents[1]
        / "datacreek"
        / "analysis"
        / "poincare_recentering.py",
    )
    poincare_recentering = importlib.util.module_from_spec(spec)
    assert isinstance(spec.loader, importlib.abc.Loader)
    spec.loader.exec_module(poincare_recentering)
    recenter_embeddings = poincare_recentering.recenter_embeddings

if np is not None and poincare_recentering.torch is not None:
    torch = poincare_recentering.torch
else:  # pragma: no cover - torch optional
    torch = None


def test_recenter_embeddings_returns_origin_mean():
    if np is None:
        pytest.skip("numpy not installed")
    embs = {0: np.array([0.1, 0.0]), 1: np.array([-0.1, 0.0])}
    rec = recenter_embeddings(embs)
    vals = np.stack(list(rec.values()))
    center = vals.mean(axis=0)
    assert np.linalg.norm(center) < 1e-3
    assert all(np.linalg.norm(v) < 1 - 1e-6 for v in vals)


def test_exp_log_inverse():
    if np is None:
        pytest.skip("numpy not installed")
    x = np.array([0.2, -0.1])
    v = poincare_recentering._log_map(np.zeros(2), x)
    y = poincare_recentering._exp_map_zero(v)
    assert np.linalg.norm(y - x) < 1e-6


def test_exp_log_grad_fp16():
    if torch is None:
        pytest.skip("torch not installed")
    x = torch.tensor([0.2, 0.1], dtype=torch.float16, requires_grad=True)
    v = poincare_recentering._log_map_zero_torch(x)
    y = poincare_recentering._exp_map_zero_torch(v)
    jac = torch.autograd.functional.jacobian(
        lambda z: poincare_recentering._exp_map_zero_torch(
            poincare_recentering._log_map_zero_torch(z)
        ),
        x,
    )
    assert torch.allclose(jac, torch.eye(2, dtype=torch.float16), rtol=1e-3, atol=1e-3)


def test_measure_overshoot_reduces_with_clamp():
    if np is None:
        pytest.skip("numpy not installed")
    radii = [0.5, 1.0, 1.5]
    res = poincare_recentering.measure_overshoot(radii, kappa=-1.0, num_samples=32)
    assert all(a <= b for a, b in zip(res["clamp"], res["noclamp"]))


def test_exp_log_inverse_precisions():
    if np is None:
        pytest.skip("numpy not installed")
    x32 = np.array([0.3, -0.2], dtype=np.float32)
    v32 = poincare_recentering._log_map(np.zeros(2, dtype=np.float32), x32)
    y32 = poincare_recentering._exp_map_zero(v32, delta=1e-6)
    assert np.linalg.norm(y32 - x32) < 1e-6
    if torch is not None:
        x16 = torch.tensor([0.3, 0.2], dtype=torch.float16)
        v16 = poincare_recentering._log_map_zero_torch(x16)
        y16 = poincare_recentering._exp_map_zero_torch(v16, delta=1e-6)
        assert torch.allclose(y16, x16, atol=1e-3, rtol=1e-3)


def test_exp_map_clamps_to_ball():
    if np is None:
        pytest.skip("numpy not installed")
    v = np.array([10.0, 0.0])
    y = poincare_recentering._exp_map_zero(v)
    assert np.linalg.norm(y) <= 1 - 1e-6


def test_recentering_fp16_robustness(monkeypatch):
    if np is None:
        pytest.skip("numpy not installed")
    rng = np.random.default_rng(0)
    curvatures = [-1.0, -0.5, -2.0]
    deltas_algo = []
    deltas_naive = []
    for c in curvatures:
        r = 0.8
        eucl = np.tanh(0.5 * (abs(c) ** 0.5) * r)
        dirs = rng.standard_normal((1000, 2))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        pts = dirs * eucl
        rec = poincare_recentering.recenter_embeddings(
            {i: pts[i] for i in range(len(pts))}, curvature=abs(c)
        )
        outs = np.vstack([rec[i] for i in range(len(pts))])
        radii_out = [
            poincare_recentering.hyperbolic_radius(o.astype(float), c=abs(c))
            for o in outs
        ]
        deltas_algo.append(np.median(np.abs(np.array(radii_out) - r)))
        center = pts.mean(axis=0)
        naive = pts - center
        radii_naive = [
            poincare_recentering.hyperbolic_radius(n, c=abs(c)) for n in naive
        ]
        deltas_naive.append(np.median(np.abs(np.array(radii_naive) - r)))
        assert not np.isnan(outs).any()
    assert np.median(deltas_algo) <= np.median(deltas_naive) * 1.1


def test_trace_overshoot_parquet(tmp_path):
    if np is None:
        pytest.skip("numpy not installed")
    try:
        import pyarrow.parquet as pq
    except Exception:
        pytest.skip("pyarrow not installed")

    out = tmp_path / "overshoot.parquet"
    poincare_recentering.trace_overshoot_parquet(str(out), num_points=10)

    table = pq.read_table(out)
    assert table.num_rows == 30
