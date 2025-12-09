import networkx as nx
import numpy as np
import pytest

from datacreek.analysis import graphwave_bandwidth as gb


def test_estimate_lambda_max_empty_graph():
    empty = nx.Graph()
    assert gb.estimate_lambda_max(empty) == 0.0


def test_estimate_lambda_max_path_graph():
    g = nx.path_graph(4)
    lmax = gb.estimate_lambda_max(g, iters=10)
    # random initialization can yield slightly lower estimates
    assert 1.5 < lmax <= 2.1


def test_update_graphwave_bandwidth_updates_and_caches():
    """Update logic should be deterministic with a fixed RNG seed."""

    np.random.seed(0)
    g = nx.path_graph(4)
    t1 = gb.update_graphwave_bandwidth(g, threshold=0.0, iters=5)
    assert pytest.approx(t1, rel=1e-6) == g.graph["gw_t"]

    np.random.seed(0)
    t2 = gb.update_graphwave_bandwidth(g, threshold=0.1, iters=5)
    assert pytest.approx(t2, rel=1e-6) == t1

    np.random.seed(0)
    g.add_edge(0, 3)
    t3 = gb.update_graphwave_bandwidth(g, threshold=0.0, iters=5)
    assert t3 != pytest.approx(t1, rel=1e-6)


class DummySp:
    def diags(self, d):
        return np.diag(d)

    def eye(self, n, format=None):
        return np.eye(n)


def test_estimate_lambda_max_with_scipy_monkeypatch(monkeypatch):
    g = nx.cycle_graph(3)
    monkeypatch.setattr(gb, "sp", DummySp(), raising=False)
    monkeypatch.setattr(
        nx,
        "to_scipy_sparse_array",
        lambda graph, format="csr": nx.to_numpy_array(graph),
    )
    lmax = gb.estimate_lambda_max(g, iters=3)
    assert lmax > 0


def test_estimate_lambda_max_without_scipy(monkeypatch):
    g = nx.cycle_graph(4)
    monkeypatch.setattr(gb, "sp", None, raising=False)
    lmax = gb.estimate_lambda_max(g, iters=3)
    assert lmax > 0


def test_update_graphwave_bandwidth_no_update():
    g = nx.path_graph(3)
    t1 = gb.update_graphwave_bandwidth(g, threshold=0.0, iters=2)
    t2 = gb.update_graphwave_bandwidth(g, threshold=1.0, iters=2)
    assert t2 == t1
