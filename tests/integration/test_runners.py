import types

import networkx as nx

import datacreek.analysis.monitoring
import datacreek.core.runners as runners
from datacreek.core.knowledge_graph import KnowledgeGraph
from datacreek.core.runners import GraphWaveRunner, Node2VecRunner


def test_node2vec_runner_uses_config(monkeypatch):
    kg = KnowledgeGraph()
    kg.graph.add_edge("a", "b")
    captured = {}

    def fake_compute(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(kg, "compute_node2vec_embeddings", fake_compute)
    monkeypatch.setattr(
        runners,
        "load_config",
        lambda: {"embeddings": {"node2vec": {"p": 3.0, "q": 4.0, "d": 16}}},
    )
    runner = Node2VecRunner(kg)
    runner.run()
    assert captured["p"] == 3.0
    assert captured["q"] == 4.0
    assert captured["dimensions"] == 16


def test_node2vec_runner_sets_var_norm(monkeypatch):
    kg = KnowledgeGraph()
    kg.graph.add_edge("a", "b")

    def fake_compute(**kwargs):
        kg.graph.nodes["a"]["embedding"] = [1.0, 0.0]
        kg.graph.nodes["b"]["embedding"] = [0.0, 1.0]

    monkeypatch.setattr(kg, "compute_node2vec_embeddings", fake_compute)
    monkeypatch.setattr(
        runners, "load_config", lambda: {"embeddings": {"node2vec": {}}}
    )
    calls = []
    monkeypatch.setattr(
        datacreek.analysis.monitoring,
        "update_metric",
        lambda n, v: calls.append((n, v)),
    )
    runner = Node2VecRunner(kg)
    runner.run()
    assert "var_norm" in kg.graph.graph
    assert abs(kg.graph.graph["var_norm"]) < 1e-6
    assert any(n == "n2v_var_norm" for n, _ in calls)


def test_graphwave_runner_sets_entropy(monkeypatch):
    kg = KnowledgeGraph()
    kg.graph.add_edge("a", "b")
    monkeypatch.setattr(kg, "compute_graphwave_embeddings", lambda *a, **k: None)
    monkeypatch.setattr(kg, "graphwave_entropy", lambda: 0.5)
    calls = []
    monkeypatch.setattr(
        datacreek.analysis.monitoring,
        "update_metric",
        lambda n, v: calls.append((n, v)),
    )
    runner = GraphWaveRunner(kg)
    runner.run([1.0])
    assert kg.graph.graph["gw_entropy"] == 0.5
    assert any(n == "gw_entropy" for n, _ in calls)


def test_poincare_runner_recenters(monkeypatch):
    kg = KnowledgeGraph()
    kg.graph.add_edge("a", "b")

    import numpy as np

    monkeypatch.setattr(
        "datacreek.analysis.fractal.poincare_embedding",
        lambda *a, **k: {"a": np.array([0.95, 0.0]), "b": np.array([0.96, 0.0])},
    )
    runner = runners.PoincareRunner(kg)
    runner.run()
    import numpy as np

    norms = [
        np.linalg.norm(kg.graph.nodes[n]["poincare_embedding"]) for n in kg.graph.nodes
    ]
    assert max(norms) <= 0.81


def test_graphwave_runner_components(monkeypatch):
    kg = KnowledgeGraph()
    for i in range(1002):
        kg.graph.add_node(f"a{i}")
    for i in range(1001):
        kg.graph.add_edge(f"a{i}", f"a{i+1}")
    for i in range(1002, 2004):
        kg.graph.add_node(f"b{i}")
    for i in range(1002, 2003):
        kg.graph.add_edge(f"b{i}", f"b{i+1}")
    calls = []
    import numpy as np

    monkeypatch.setattr(
        "datacreek.analysis.fractal.graphwave_embedding_chebyshev",
        lambda g, *a, **k: (
            {n: np.zeros(1) for n in g.nodes()}
            if not calls.append(len(g))
            else {n: np.zeros(1) for n in g.nodes()}
        ),
    )
    monkeypatch.setattr(kg, "graphwave_entropy", lambda: 0.0)
    runner = GraphWaveRunner(kg)
    runner.run([1.0], num_points=4, order=3)
    assert len(calls) == 2
