import networkx as nx
import numpy as np
import pytest

from datacreek.analysis import graphwave_bandwidth as gw


@pytest.mark.heavy
def test_estimate_lambda_max_positive():
    g = nx.path_graph(4)
    lmax = gw.estimate_lambda_max(g, iters=3)
    assert lmax > 0


@pytest.mark.heavy
def test_estimate_lambda_max_with_scipy(monkeypatch):
    class DummySP:
        @staticmethod
        def diags(v):
            return np.diag(np.asarray(v))

        @staticmethod
        def eye(n, format="csr"):
            return np.eye(n)

    g = nx.path_graph(3)

    monkeypatch.setattr(gw, "sp", DummySP)
    monkeypatch.setattr(
        nx, "to_scipy_sparse_array", lambda *a, **k: nx.to_numpy_array(*a)
    )
    lmax = gw.estimate_lambda_max(g, iters=2)
    assert lmax > 0


@pytest.mark.heavy
def test_update_graphwave_bandwidth_changes():
    g = nx.cycle_graph(4)
    first = gw.update_graphwave_bandwidth(g, threshold=0.0, iters=2)
    # re-running should not change value significantly
    second = gw.update_graphwave_bandwidth(g, threshold=1.0, iters=2)
    assert pytest.approx(first) == second
    # forced recompute when threshold=0
    third = gw.update_graphwave_bandwidth(g, threshold=0.0, iters=2)
    assert g.graph["gw_t"] == pytest.approx(third)
