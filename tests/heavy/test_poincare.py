import importlib
import sys
import types

import numpy as np

import datacreek.analysis.poincare_recentering as pr


def test_recenter_and_hyperbolic(monkeypatch):
    embeddings = {i: np.array([0.2 * i, 0.0]) for i in range(4)}
    rec = pr.recenter_embeddings(embeddings, curvature=1.0)
    center = np.mean(list(rec.values()), axis=0)
    assert np.linalg.norm(center) < 0.1
    curves = pr.measure_overshoot([0.1, 0.2], num_samples=4)
    assert set(curves) == {"clamp", "noclamp"}
    assert len(curves["clamp"]) == 2


def test_trace_overshoot_parquet(monkeypatch, tmp_path):
    captured = {}
    fake_pa = types.ModuleType("pyarrow")
    fake_pa.table = lambda d: d
    fake_pq = types.ModuleType("pyarrow.parquet")

    def write_table(tbl, path):
        captured["rows"] = len(tbl["kappa"])
        captured["path"] = path

    fake_pq.write_table = write_table
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pa)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", fake_pq)
    importlib.reload(pr)
    out = tmp_path / "out.parquet"
    pr.trace_overshoot_parquet(str(out), num_points=3, curvatures=(-1.0,))
    assert captured["path"] == str(out)
