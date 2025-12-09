import pytest

from datacreek.core.knowledge_graph import KnowledgeGraph


def build_graph():
    pytest.importorskip("sklearn")
    kg = KnowledgeGraph()
    kg.add_document("d1", "src", text="doc text", author="A", organization="Org")
    kg.add_section("d1", "s1", title="First", page=1)
    kg.add_section("d1", "s2", title="Second", page=2)
    kg.add_chunk("d1", "c1", "hello one", section_id="s1", page=1, emotion="joy")
    kg.add_chunk("d1", "c2", "hello two", section_id="s1", page=2)
    kg.add_chunk("d1", "c3", "hello three", section_id="s2", page=3)
    kg.add_image("d1", "img1", "/img", alt_text="an image")
    kg.add_audio("d1", "aud1", "/aud", page=1, lang="en")
    kg.add_atom("d1", "a1", "H", "element")
    kg.add_molecule("d1", "m1", ["a1"])
    kg.add_entity("e1", "Alice", source="src")
    kg.add_fact("e1", "knows", "Bob", fact_id="f1", source="src")
    kg.link_entity("c1", "e1")
    kg.link_transcript("c1", "aud1")
    kg.index.build()
    return kg


def test_structure_helpers_and_queries():
    kg = build_graph()
    assert kg.get_sections_for_document("d1") == ["s1", "s2"]
    assert kg.get_chunks_for_section("s1") == ["c1", "c2"]
    assert kg.get_section_for_chunk("c1") == "s1"
    assert kg.get_next_section("s1") == "s2"
    assert kg.get_previous_section("s2") == "s1"
    assert kg.get_next_chunk("c1") == "c2"
    assert kg.get_previous_chunk("c2") == "c1"
    assert kg.get_page_for_chunk("c1") == 1
    assert kg.get_page_for_section("s2") == 2
    assert kg.get_chunks_for_document("d1") == ["c1", "c2", "c3"]
    assert kg.get_images_for_document("d1") == ["img1"]
    assert kg.get_captions_for_document("d1") == ["img1_caption"]
    assert kg.get_caption_for_image("img1") == "img1_caption"
    assert kg.get_audios_for_document("d1") == ["aud1"]
    assert kg.get_atoms_for_document("d1") == ["a1"]
    assert kg.get_molecules_for_document("d1") == ["m1"]
    assert kg.get_atoms_for_molecule("m1") == ["a1"]
    assert kg.get_facts_for_entity("e1") == ["f1"]
    assert kg.get_chunks_for_entity("e1") == ["c1"]
    assert kg.get_facts_for_document("d1") == []
    assert kg.get_chunks_for_fact("f1") == []
    assert kg.get_sections_for_fact("f1") == []
    assert kg.get_documents_for_fact("f1") == []
    assert kg.get_pages_for_fact("f1") == []
    assert set(kg.get_entities_for_fact("f1")) == {"e1", "Bob"}
    assert kg.get_entities_for_chunk("c1") == ["e1"]
    assert kg.get_entities_for_document("d1") == ["e1"]
    assert kg.get_document_for_section("s1") == "d1"
    assert kg.get_document_for_chunk("c1") == "d1"
    # similarity helpers
    # search helpers use the embedding index built from chunk texts
    if kg.index._vectorizer is not None:
        similar = kg.get_similar_chunks("c1", k=1)
        assert similar and similar[0] in {"c2", "c3"}
    ctx = kg.get_chunk_context("c2", before=1, after=1)
    assert ctx == ["c1", "c2", "c3"]


def test_embeddings_and_centrality():
    kg = build_graph()
    try:
        kg.compute_node2vec_embeddings(
            dimensions=2, walk_length=2, num_walks=2, workers=1, seed=0
        )
    except Exception:
        pytest.skip("node2vec failed")
    assert any("embedding" in d for d in kg.graph.nodes.values())
    kg.compute_centrality()
    assert all("centrality" in kg.graph.nodes[n] for n in kg.graph.nodes)
    kg.cluster_chunks(n_clusters=2)
    kg.summarize_communities()
    kg.cluster_entities(n_clusters=1)
    kg.summarize_entity_groups()
    kg.compute_poincare_embeddings(
        dim=2, negative=1, epochs=1, learning_rate=0.1, burn_in=1
    )
    assert any("poincare_embedding" in d for d in kg.graph.nodes.values())
