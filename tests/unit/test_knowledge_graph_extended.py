import pytest

from datacreek.core.knowledge_graph import KnowledgeGraph


def build_sample_graph():
    kg = KnowledgeGraph()
    kg.add_document("d1", "src", author="author", organization="org")
    kg.add_section("d1", "s1", title="Intro")
    kg.add_chunk("d1", "c1", "hello world", section_id="s1")
    kg.add_chunk("d1", "c2", "hello again", section_id="s1")
    kg.add_document("d2", "src2")
    kg.add_section("d2", "s2", title="Other")
    kg.add_chunk("d2", "c3", "other text", section_id="s2")
    kg.add_entity("e1", "Entity")
    kg.add_entity("e2", "Other")
    kg.link_entity("c1", "e1")
    kg.link_entity("c2", "e1")
    kg.link_entity("c3", "e1")
    kg.graph.add_edge("c1", "c2", relation="similar_to")
    kg.index.build()
    return kg


def test_linking_helpers_and_conversions():
    kg = build_sample_graph()
    assert kg.link_chunks_by_entity() >= 1
    assert kg.link_documents_by_entity() >= 1
    assert kg.link_sections_by_entity() >= 1
    assert kg.link_authors_organizations() >= 1
    data = kg.to_dict()
    rebuilt = KnowledgeGraph.from_dict(data)
    assert set(rebuilt.graph.nodes) == set(kg.graph.nodes)
    assert set(rebuilt.graph.edges) >= set(kg.graph.edges)
    text = rebuilt.to_text()
    assert "hello" in text


def test_search_with_links_and_queries():
    kg = build_sample_graph()
    kg.index.build()
    if kg.index._vectorizer is None:
        pytest.skip("sklearn missing")
    results = kg.search_with_links("hello", k=1, hops=1)
    assert "c2" in results
    data = kg.search_with_links_data("hello", k=1, hops=1)
    ids = [item["id"] for item in data]
    assert "c2" in ids


def test_chunk_document_queries_and_cleanup():
    kg = build_sample_graph()
    assert kg.chunks_by_emotion("joy") == []
    assert kg.chunks_by_modality("text") == []
    kg.graph.nodes["c1"]["emotion"] = "joy"
    kg.graph.nodes["c2"]["modality"] = "text"
    assert kg.chunks_by_emotion("joy") == ["c1"]
    assert kg.chunks_by_modality("text") == ["c2"]
    kg.graph.nodes["c1"]["birth_date"] = "2023-05-01T00:00:00"
    assert kg.normalize_date_fields() == 1
    kg.remove_chunk("c2")
    assert "c2" not in kg.graph
    kg.remove_document("d2")
    assert "d2" not in kg.graph
