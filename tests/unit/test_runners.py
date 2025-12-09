import importlib

import networkx as nx
import numpy as np
import pytest

import datacreek.core.runners as runners


class FakeGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.graph.add_nodes_from([1, 2])
        self.graph.graph = {}

    def compute_node2vec_embeddings(self, **kwargs):
        self.node2vec_args = kwargs
        self.graph.nodes[1]["embedding"] = [1.0, 0.0]
        self.graph.nodes[2]["embedding"] = [0.0, 1.0]

    def compute_graphwave_embeddings(self, **kwargs):
        self.gw_args = kwargs
        for n in self.graph.nodes:
            self.graph.nodes[n]["graphwave_embedding"] = [0.1, 0.1]

    def graphwave_entropy(self):
        return 0.5


def test_node2vec_runner(monkeypatch):
    g = FakeGraph()
    monkeypatch.setattr(
        runners,
        "load_config",
        lambda: {"embeddings": {"node2vec": {"d": 64, "p": 1.5, "q": 0.5}}},
    )
    runner = runners.Node2VecRunner(g)
    runner.run()
    assert g.node2vec_args["dimensions"] == 64
    assert g.graph.graph["var_norm"] == pytest.approx(0.0)


def test_graphwave_runner(monkeypatch):
    g = FakeGraph()
    g.graph.add_edge(1, 2)
    gbw = importlib.import_module("datacreek.analysis.graphwave_bandwidth")
    mon = importlib.import_module("datacreek.analysis.monitoring")
    frac = importlib.import_module("datacreek.analysis.fractal")
    monkeypatch.setattr(gbw, "update_graphwave_bandwidth", lambda G: 2.0)
    monkeypatch.setattr(mon, "gw_entropy", None, raising=False)
    monkeypatch.setattr(mon, "update_metric", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(
        frac,
        "graphwave_embedding_chebyshev",
        lambda sub, scales, num_points, order: {
            n: np.array([0.2, 0.2]) for n in sub.nodes
        },
    )
    runner = runners.GraphWaveRunner(g)
    runner.run(scales=[2.0])
    assert g.gw_args["scales"] == [2.0]
    assert g.graph.graph["gw_entropy"] == 0.5


def test_poincare_runner(monkeypatch):
    g = FakeGraph()
    g.graph.add_edge(1, 2)
    frac = importlib.import_module("datacreek.analysis.fractal")
    monkeypatch.setattr(
        frac,
        "poincare_embedding",
        lambda G, **k: {1: np.array([1.0, 0.0]), 2: np.array([1.0, 0.0])},
    )
    runner = runners.PoincareRunner(g)
    runner.run()
    v = g.graph.nodes[1]["poincare_embedding"]
    assert pytest.approx(np.linalg.norm(v)) == pytest.approx(0.8)
