import networkx as nx
import numpy as np

from datacreek.analysis.graphwave_bandwidth import (
    estimate_lambda_max,
    update_graphwave_bandwidth,
)
from datacreek.core.knowledge_graph import KnowledgeGraph
from datacreek.core.runners import GraphWaveRunner


def test_estimate_lambda_max_cycle():
    g = nx.cycle_graph(4)
    A = nx.to_numpy_array(g)
    deg = A.sum(axis=1)
    with np.errstate(divide="ignore"):
        d_isqrt = 1.0 / np.sqrt(deg)
    d_isqrt[~np.isfinite(d_isqrt)] = 0.0
    L = np.eye(A.shape[0]) - d_isqrt[:, None] * A * d_isqrt
    eig = np.linalg.eigvalsh(L).max()
    est = estimate_lambda_max(g, iters=5)
    assert abs(est - eig) / eig < 0.1


def test_update_graphwave_bandwidth_changes():
    g = nx.path_graph(4)
    t1 = update_graphwave_bandwidth(g)
    l1 = g.graph["gw_lambda_max"]
    g.add_edge(0, 3)
    t2 = update_graphwave_bandwidth(g)
    l2 = g.graph["gw_lambda_max"]
    if abs(l2 - l1) / l1 > 0.05:
        assert t1 != t2
    else:
        assert t1 == t2


def test_graphwave_runner_dynamic(monkeypatch):
    kg = KnowledgeGraph()
    kg.graph.add_edge("a", "b")
    monkeypatch.setattr(kg, "compute_graphwave_embeddings", lambda *a, **k: None)
    monkeypatch.setattr(kg, "graphwave_entropy", lambda: 0.5)
    runner = GraphWaveRunner(kg)
    runner.run(dynamic=True)
    assert "gw_t" in kg.graph.graph
