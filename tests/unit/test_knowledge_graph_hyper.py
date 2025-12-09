import types

import numpy as np

from datacreek.core.knowledge_graph import KnowledgeGraph


def build_kg():
    kg = KnowledgeGraph()
    kg.add_document("d", "src")
    kg.add_chunk("d", "c1", "one")
    kg.add_chunk("d", "c2", "two")
    kg.graph.nodes["c1"]["hyperbolic_embedding"] = [0.1]
    kg.graph.nodes["c2"]["hyperbolic_embedding"] = [0.2]
    kg.graph.nodes["c1"]["poincare_embedding"] = [0.1]
    kg.graph.nodes["c2"]["poincare_embedding"] = [0.2]
    kg.graph.add_node("he1", type="hyperedge")
    kg.graph.add_edge("he1", "c1")
    kg.graph.add_edge("he1", "c2")
    return kg


def test_hyperbolic_reasoning_helpers(monkeypatch):
    kg = build_kg()
    # patch analysis functions to deterministic stubs
    monkeypatch.setattr(
        "datacreek.analysis.fractal.hyperbolic_nearest_neighbors",
        lambda emb, k=5: {"c1": [("c2", 0.5)]},
    )
    assert kg.hyperbolic_neighbors("c1", k=1) == [("c2", 0.5)]

    monkeypatch.setattr(
        "datacreek.analysis.fractal.hyperbolic_reasoning",
        lambda emb, start, goal, max_steps=5: ["c1", "c2"],
    )
    assert kg.hyperbolic_reasoning("c1", "c2") == ["c1", "c2"]

    monkeypatch.setattr(
        "datacreek.analysis.fractal.hyperbolic_hypergraph_reasoning",
        lambda emb, hypers, start, goal, penalty=1.0, max_steps=5: ["c1", "he1", "c2"],
    )
    assert kg.hyperbolic_hypergraph_reasoning("c1", "c2") == ["c1", "he1", "c2"]

    monkeypatch.setattr(
        "datacreek.analysis.fractal.hyperbolic_multi_curvature_reasoning",
        lambda embs, start, goal, weights=None, max_steps=5: ["c1", "c2"],
    )
    kg.graph.nodes["c1"]["hyperbolic_embedding_0.1"] = [0.1]
    kg.graph.nodes["c2"]["hyperbolic_embedding_0.1"] = [0.2]
    assert kg.hyperbolic_multi_curvature_reasoning("c1", "c2", curvatures=[0.1]) == [
        "c1",
        "c2",
    ]


def test_fractal_levels_and_dimensions(monkeypatch):
    kg = build_kg()
    monkeypatch.setattr(
        "datacreek.analysis.fractal.build_fractal_hierarchy",
        lambda g, r, max_levels=5: [(g, {n: i for i, n in enumerate(g)}, 1)],
    )
    kg.annotate_fractal_levels([1])
    assert kg.graph.nodes["c1"]["fractal_level"] == 1

    monkeypatch.setattr(
        "datacreek.analysis.fractal.build_mdl_hierarchy",
        lambda g, r, max_levels=5: [(g, {n: i for i, n in enumerate(g)}, 1)],
    )
    kg.annotate_mdl_levels([1])
    assert kg.graph.nodes["c2"]["fractal_level"] == 1

    monkeypatch.setattr(
        "datacreek.analysis.fractal.embedding_box_counting_dimension",
        lambda coords, radii: (1.5, [(1, 2)]),
    )
    monkeypatch.setattr(
        KnowledgeGraph, "box_counting_dimension", lambda self, r: (2.0, [(1, 2)])
    )
    assert kg.dimension_distortion([1]) == 0.5
    dim, counts = kg.embedding_box_counting_dimension("hyperbolic_embedding", [1])
    assert dim == 1.5 and counts == [(1, 2)]


def test_detect_automorphisms(monkeypatch):
    kg = build_kg()
    monkeypatch.setattr(
        "datacreek.analysis.symmetry.automorphisms",
        lambda g, max_count=10: [{"c1": "c2"}],
    )
    assert kg.detect_automorphisms() == [{"c1": "c2"}]


def test_automorphism_group_and_quotient(monkeypatch):
    kg = build_kg()
    monkeypatch.setattr(
        "datacreek.analysis.symmetry.automorphism_group_order",
        lambda g, max_count=100: 4,
    )
    order = kg.automorphism_group_order(max_count=5)
    assert order == 4

    monkeypatch.setattr(
        "datacreek.analysis.symmetry.automorphism_orbits",
        lambda g, max_count=10: [{"c1", "c2"}],
    )
    import networkx as nx

    q_graph = nx.Graph()
    q_graph.add_edge("0", "1")
    monkeypatch.setattr(
        "datacreek.analysis.symmetry.quotient_graph",
        lambda g, orbits: (q_graph, {"c1": 0, "c2": 1}),
    )
    q, mapping = kg.quotient_by_symmetry(max_count=5)
    assert mapping == {"c1": 0, "c2": 1}
    assert list(q.edges()) == [("0", "1")]


# New tests


def test_average_radius_and_explain_node(monkeypatch):
    kg = build_kg()
    # two nodes with Poincar\xe9 embeddings
    kg.graph.nodes["c1"]["poincare_embedding"] = [1.0]
    kg.graph.nodes["c2"]["poincare_embedding"] = [1.0]
    monkeypatch.setattr(
        "datacreek.analysis.governance.average_hyperbolic_radius",
        lambda emb: sum(float(v[0]) for v in emb.values()) / len(emb),
    )
    assert kg.average_hyperbolic_radius() == 1.0

    # prepare simple link and embeddings for attention heatmap
    kg.graph.add_edge("c1", "c2")
    kg.graph.nodes["c1"]["embedding"] = [1.0, 0.0]
    kg.graph.nodes["c2"]["embedding"] = [0.0, 1.0]
    monkeypatch.setattr(
        "datacreek.analysis.hypergraph.hyperedge_attention_scores",
        lambda edges, feats: [0.5] * len(edges),
    )
    result = kg.explain_node("c1", hops=1)
    assert "c2" in result["nodes"]
    assert result["attention"].get("c1->c2") == 0.5
