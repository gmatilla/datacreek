import types

import numpy as np
import pytest

from datacreek.core.knowledge_graph import KnowledgeGraph


def build_complex_graph():
    kg = KnowledgeGraph()
    kg.add_document("d1", "src1")
    kg.add_section("d1", "s1", title="Section1")
    kg.add_section("d1", "s2", title="Section2")
    kg.add_chunk("d1", "c1", "foo", section_id="s1")
    kg.add_chunk("d1", "c2", "bar", section_id="s2")
    # adding image with alt_text automatically creates a caption node
    kg.add_image("d1", "img1", "/i1", alt_text="cap")
    kg.add_audio("d1", "a1", "/a1", lang="en")
    kg.add_atom("d1", "at1", "A", "t")
    kg.add_molecule("d1", "m1", ["at1"])
    kg.add_hyperedge("he1", ["c1", "c2"])  # small hyperedge
    kg.graph.nodes["c1"]["hyperbolic_embedding"] = [0.1]
    kg.graph.nodes["c2"]["hyperbolic_embedding"] = [0.2]
    return kg


def test_navigation_and_lookup_helpers():
    kg = build_complex_graph()
    assert kg.get_sections_for_document("d1") == ["s1", "s2"]
    assert kg.get_chunks_for_section("s1") == ["c1"]
    assert kg.get_chunks_for_document("d1") == ["c1", "c2"]
    assert kg.get_section_for_chunk("c2") == "s2"
    kg.graph.edges["s1", "s2"]["relation"] = "next_section"
    assert kg.get_next_section("s1") == "s2"
    assert kg.get_previous_section("s2") == "s1"
    kg.graph.edges["c1", "c2"]["relation"] = "next_chunk"
    assert kg.get_next_chunk("c1") == "c2"
    assert kg.get_previous_chunk("c2") == "c1"
    kg.graph.nodes["c1"]["page"] = 5
    kg.graph.nodes["s1"]["page"] = 3
    assert kg.get_page_for_chunk("c1") == 5
    assert kg.get_page_for_section("s1") == 3
    assert kg.get_images_for_document("d1") == ["img1"]
    assert kg.get_captions_for_document("d1") == ["img1_caption"]
    assert kg.get_caption_for_image("img1") == "img1_caption"
    assert kg.get_audios_for_document("d1") == ["a1"]
    assert kg.get_atoms_for_document("d1") == ["at1"]
    assert kg.get_molecules_for_document("d1") == ["m1"]


def test_fractal_and_hyperbolic_helpers(monkeypatch):
    kg = build_complex_graph()
    # patch fractal helpers to return simple values
    calls = [0]

    def fake_cov(g):
        calls[0] += 1
        return 0.0 if calls[0] == 1 else 0.5

    monkeypatch.setattr("datacreek.analysis.fractal.fractal_level_coverage", fake_cov)
    monkeypatch.setattr(
        "datacreek.analysis.fractal.diversification_score",
        lambda g_full, sub, radii, max_dim=1, dimension=0: 0.5,
    )
    monkeypatch.setattr(
        "datacreek.analysis.fractal.hyperbolic_nearest_neighbors",
        lambda emb, k=5: {"c1": [("c2", 0.3)]},
    )
    monkeypatch.setattr(
        "datacreek.analysis.fractal.hyperbolic_reasoning",
        lambda emb, s, g, max_steps=5: [s, g],
    )
    monkeypatch.setattr(
        "datacreek.analysis.fractal.hyperbolic_hypergraph_reasoning",
        lambda emb, he, s, g, penalty=1.0, max_steps=5: [s, g],
    )
    monkeypatch.setattr(
        "datacreek.analysis.hypergraph.hyper_sagnn_embeddings",
        lambda members, feats, embed_dim=None, seed=None: np.zeros((1, feats.shape[1])),
    )
    monkeypatch.setattr(
        "datacreek.analysis.fractal.mdl_optimal_radius", lambda counts: 0
    )
    # coverage for fractal_coverage and ensure_fractal_coverage
    assert kg.fractal_coverage() == 0.0
    cov = kg.ensure_fractal_coverage(0.5, [1])
    assert cov == 0.5
    # diversification and hyperbolic helpers
    assert kg.diversification_score(["c1", "c2"], [1]) == 0.5
    assert kg.select_diverse_nodes(["c1", "c2"], 1, [1]) == ["c1"]
    assert kg.hyperbolic_neighbors("c1", k=1) == [("c2", 0.3)]
    assert kg.hyperbolic_reasoning("c1", "c2") == ["c1", "c2"]
    assert kg.hyperbolic_hypergraph_reasoning("c1", "c2") == ["c1", "c2"]
    # predict hyperedges uses hyper_sagnn embeddings
    import sys
    import types

    sp = types.SimpleNamespace(
        cosine_similarity=lambda x: np.array([[1.0, 0.9], [0.9, 1.0]])
    )
    sk = types.SimpleNamespace(metrics=types.SimpleNamespace(pairwise=sp))
    monkeypatch.setitem(sys.modules, "sklearn", sk)
    monkeypatch.setitem(sys.modules, "sklearn.metrics", sk.metrics)
    monkeypatch.setitem(sys.modules, "sklearn.metrics.pairwise", sp)
    preds = kg.predict_hyperedges(k=1, threshold=0.5)
    assert isinstance(preds, list)
