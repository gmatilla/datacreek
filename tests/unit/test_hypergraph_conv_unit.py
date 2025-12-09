import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root on sys.path for direct module imports
sys.path.append(str(Path(__file__).resolve().parents[2]))


def test_lambda_max_cache(monkeypatch):
    module = importlib.reload(
        importlib.import_module("datacreek.analysis.hypergraph_conv")
    )
    # patch RNG to track computations
    call_count = {"n": 0}

    original_rng = np.random.default_rng

    def fake_rng(seed):
        call_count["n"] += 1
        return original_rng(seed)

    monkeypatch.setattr(module.np.random, "default_rng", fake_rng)
    module.lambda_max_cache.clear()

    B = np.array([[1, 0], [1, 1], [0, 1]])
    L = module.hypergraph_laplacian(B)

    module.estimate_lambda_max(L, g_id="g", num_edges=2)
    assert call_count["n"] == 1
    # same edge count → cached
    module.estimate_lambda_max(L, g_id="g", num_edges=2)
    assert call_count["n"] == 1
    # edge count varies >5% → recompute
    module.estimate_lambda_max(L, g_id="g", num_edges=3)
    assert call_count["n"] == 2


def test_adaptive_K_logging(caplog):
    module = importlib.reload(
        importlib.import_module("datacreek.analysis.hypergraph_conv")
    )
    B = np.array([[1, 0], [1, 1], [0, 1]])
    L = module.hypergraph_laplacian(B)
    X = np.eye(3)

    lamb_max = module.estimate_lambda_max(L)
    caplog.set_level("INFO")
    module.chebyshev_conv(X, L, K=None, lambda_k=lamb_max - 0.5, g_id="g2", num_edges=2)
    assert any("spec_K_chosen=3" in r.getMessage() for r in caplog.records)
