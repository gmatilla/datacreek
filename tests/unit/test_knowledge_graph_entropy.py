import sys
import types

import networkx as nx
import numpy as np

from datacreek.core.knowledge_graph import KnowledgeGraph


def build_graph():
    kg = KnowledgeGraph()
    kg.add_document("d1", "src")
    kg.add_chunk("d1", "c1", "one")
    kg.add_chunk("d1", "c2", "two")
    kg.graph.add_edge("c1", "c2", weight=1.0)
    kg.graph.nodes["c1"]["embedding"] = [1.0, 0.0]
    kg.graph.nodes["c1"]["graphwave_embedding"] = [0.1]
    kg.graph.nodes["c1"]["poincare_embedding"] = [0.2]
    kg.graph.nodes["c2"]["embedding"] = [0.0, 1.0]
    kg.graph.nodes["c2"]["graphwave_embedding"] = [0.4]
    kg.graph.nodes["c2"]["poincare_embedding"] = [0.5]
    return kg


def test_entropy_and_governance():
    kg = build_graph()
    assert kg.graph_entropy() >= 0
    assert kg.subgraph_entropy(["c1"]) >= 0
    assert kg.structural_entropy(0) >= 0
    metrics = kg.governance_metrics()
    assert set(metrics) == {"alignment_corr", "hyperbolic_radius", "bias_wasserstein"}

    groups = {"c1": "a", "c2": "b"}
    rescaled = kg.mitigate_bias_wasserstein(groups)
    assert set(rescaled) == {"c1", "c2"}


def test_auto_tool_calls_all():
    kg = KnowledgeGraph()
    kg.graph.add_node("n1", text="hello one")
    kg.graph.add_node("n2", text="two")
    tools = [("up", "one"), ("down", "two")]
    res = kg.auto_tool_calls_all(tools)
    assert res["n1"].endswith("(one)]")
    assert res["n2"].endswith("(two)]")


def test_information_and_threshold(monkeypatch):
    kg = KnowledgeGraph()
    kg.add_document("d1", "src")
    kg.add_chunk("d1", "c1", "hello")
    kg.add_chunk("d1", "c2", "world")
    kg.graph.add_edge("c1", "c2", weight=2.0)
    kg.graph.nodes["c1"]["embedding"] = [1.0, 0.0]
    kg.graph.nodes["c2"]["embedding"] = [0.0, 1.0]
    labels = {"c1": 0, "c2": 1}
    dummy = types.SimpleNamespace(
        LogisticRegression=lambda max_iter=1000, n_jobs=1: types.SimpleNamespace(
            fit=lambda X, y: None,
            predict_proba=lambda X: np.ones((len(X), len(set(labels.values()))))
            / len(set(labels.values())),
        )
    )
    monkeypatch.setitem(sys.modules, "sklearn.linear_model", dummy)
    loss = kg.graph_information_bottleneck(labels)
    assert loss > 0
    thresh = kg.adaptive_triangle_threshold(weight="weight", base=2.0, scale=5.0)
    assert thresh >= 1
