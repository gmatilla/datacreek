import builtins
import importlib
import sys
import types

import numpy as np

import datacreek.analysis.poincare_recentering as pr


def test_mobius_and_radius_roundtrip():
    x = np.array([0.2, 0.1])
    y = np.array([-0.1, 0.2])
    added = pr._mobius_add(x, y)
    assert np.linalg.norm(added) < 1
    zero = pr._mobius_add(x, pr._mobius_neg(x))
    assert np.allclose(zero, np.zeros_like(x))


def test_exp_log_map_consistency():
    v = np.array([0.3, -0.2])
    pt = pr._exp_map_zero(v)
    back = pr._log_map(np.zeros_like(v), pt)
    assert np.allclose(v, back, atol=1e-7)
    assert pr.hyperbolic_radius(pt) > 0


def test_recenter_and_measure():
    emb = {i: np.array([0.2 * i, 0.0]) for i in range(1, 4)}
    rec = pr.recenter_embeddings(emb)
    for v in rec.values():
        assert np.linalg.norm(v) < 1
    curves = pr.measure_overshoot([0.2, 0.5], num_samples=8)
    assert set(curves) == {"clamp", "noclamp"}
    assert len(curves["clamp"]) == 2


def test_trace_overshoot_parquet(monkeypatch, tmp_path):
    captured = {}
    fake_pa = types.ModuleType("pyarrow")
    fake_pa.table = lambda d: d
    fake_pq = types.ModuleType("pyarrow.parquet")
    fake_pq.write_table = lambda tbl, path: captured.update(
        {"rows": len(tbl["kappa"]), "path": path}
    )
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pa)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", fake_pq)
    out = tmp_path / "o.parquet"
    pr.trace_overshoot_parquet(str(out), num_points=5, curvatures=(-1.0,))
    assert captured["path"] == str(out)
    assert captured["rows"] == 5
