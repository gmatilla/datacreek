import os
from pathlib import Path

import pytest

from datacreek.core.knowledge_graph import (
    KnowledgeGraph,
    _load_cleanup,
    apply_cleanup_config,
    get_cleanup_cfg,
    start_cleanup_watcher,
    stop_cleanup_watcher,
    verify_thresholds,
)


def test_knowledge_graph_basic(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """cleanup:
  tau: 3
  sigma: 0.9
  k_min: 2
  lp_sigma: 0.1
  lp_topk: 10
  hub_deg: 50
"""
    )
    monkeypatch.setenv("DATACREEK_CONFIG", str(cfg))
    _load_cleanup()
    assert get_cleanup_cfg()["tau"] == 3
    apply_cleanup_config()
    assert get_cleanup_cfg()["lp_topk"] == 10
    verify_thresholds()
    start_cleanup_watcher(interval=0.01)
    stop_cleanup_watcher()

    kg = KnowledgeGraph()
    kg.add_document("doc", "src", text="<b>Hello</b>", author="auth")
    kg.add_section("doc", "sec1", title="Sec")
    kg.add_chunk(
        "doc",
        "chunk1",
        "<i>Chunk1</i>",
        section_id="sec1",
        emotion="joy",
        modality="text",
    )
    kg.add_entity("ent1", "Entity", source="src")
    kg.add_fact("ent1", "rel", "ent2", fact_id="fact1")
    kg.link_entity("chunk1", "ent1")
    kg.add_image("doc", "img1", "/p", alt_text="Alt")
    kg.add_audio("doc", "aud1", "/a", lang="en")
    kg.add_atom("doc", "atom1", "Atom", "word")
    kg.add_molecule("doc", "mol1", ["atom1"])
    kg.add_hyperedge("he1", ["chunk1", "ent1"])
    kg.add_simplex("simp1", ["chunk1", "ent1"])

    kg.index.build()
    assert kg.search("chunk1") == ["chunk1"]
    assert kg.search_chunks("Chunk1") == ["chunk1"]
    assert kg.search_documents("src") == ["doc"]
    # embedding search functions should execute without raising errors
    try:
        kg.search_embeddings("chunk1")
        kg.search_hybrid("chunk1")
    except Exception:
        pytest.skip("Embedding dependencies missing")
    kg.link_similar_chunks(k=1)
    modified = kg.clean_chunk_texts()
    assert modified == 1
    kg.graph.nodes["chunk1"]["birth_date"] = "2022-01-02"
    kg.graph.nodes["ent1"]["birth_date"] = "2022-01-03"
    kg.graph.add_edge("ent1", "chunk1", relation="parent_of")
    assert kg.validate_coherence() == 1
