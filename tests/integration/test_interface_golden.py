import importlib.abc
import importlib.util
import json
from pathlib import Path

import networkx as nx
import numpy as np

spec = importlib.util.spec_from_file_location(
    "chebdiag",
    Path(__file__).resolve().parents[1]
    / "datacreek"
    / "analysis"
    / "chebyshev_diag.py",
)
chebdiag = importlib.util.module_from_spec(spec)
assert isinstance(spec.loader, importlib.abc.Loader)
spec.loader.exec_module(chebdiag)
chebyshev_diag_hutchpp = chebdiag.chebyshev_diag_hutchpp


def test_chebyshev_diag_golden():
    g = nx.path_graph(4)
    A = nx.to_numpy_array(g)
    deg = A.sum(axis=1)
    d_isqrt = 1 / np.sqrt(deg, where=deg > 0)
    d_isqrt[~np.isfinite(d_isqrt)] = 0
    L = np.eye(A.shape[0]) - d_isqrt[:, None] * A * d_isqrt
    t = 0.5
    approx = chebyshev_diag_hutchpp(
        L, t, order=5, samples=128, rng=np.random.default_rng(0)
    )
    with open(
        Path(__file__).resolve().parent / "golden" / "chebyshev_diag_path4.json"
    ) as fh:
        expected = np.array(json.load(fh))
    assert np.allclose(approx, expected, atol=1e-2)
