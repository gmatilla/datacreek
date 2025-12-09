import sys
import types

import numpy as np

from datacreek.core.knowledge_graph import KnowledgeGraph


def build_simple_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    kg.add_document("doc", "src")
    kg.add_chunk("doc", "c1", "foo")
    kg.add_chunk("doc", "c2", "bar")
    kg.add_chunk("doc", "c3", "baz")
    for node in ["c1", "c2", "c3"]:
        kg.graph.nodes[node]["embedding"] = np.array([1.0, 0.0])
        kg.graph.nodes[node]["graphwave_embedding"] = np.array([0.5, 0.0])
        kg.graph.nodes[node]["poincare_embedding"] = np.array([0.1, 0.0])
    return kg


def test_hybrid_score_and_similar(monkeypatch):
    kg = build_simple_kg()

    calls = {}

    def fake_score(a, b, gw_a, gw_b, hyp_a, hyp_b, *, gamma=0.5, eta=0.25):
        calls["args"] = (a, b, gw_a, gw_b, hyp_a, hyp_b, gamma, eta)
        return float(a[0] + b[0])

    monkeypatch.setattr(
        "datacreek.analysis.multiview.hybrid_score", fake_score, raising=False
    )

    val = kg.hybrid_score("c1", "c2")
    assert val == 2.0
    assert calls["args"][0].tolist() == [1.0, 0.0]

    ranked = kg.similar_by_hybrid("c1", k=2)
    assert ranked and ranked[0][0] == "c2"


def test_recall_at_k(monkeypatch):
    kg = build_simple_kg()

    def fake_recall(graph, queries, truth, *, k=10, gamma=0.5, eta=0.25):
        assert k == 10
        return 0.42

    monkeypatch.setitem(
        sys.modules,
        "datacreek.analysis.autotune",
        types.SimpleNamespace(recall_at_k=fake_recall),
    )

    r = kg.recall_at_k(["c1"], {"c1": ["c2"]}, k=10)
    assert r == 0.42
    assert kg.graph.graph["recall10"] == 0.42
