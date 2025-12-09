import pytest

from datacreek.core.knowledge_graph import KnowledgeGraph


def build_simple_graph():
    kg = KnowledgeGraph()
    kg.add_document("d1", "src")
    kg.add_chunk("d1", "c1", "hello world")
    kg.add_chunk("d1", "c2", "next text")
    kg.graph.add_edge("c1", "c2", relation="similar_to")
    kg.add_entity("e1", "one")
    kg.add_entity("e2", "two")
    kg.add_fact("e1", "likes", "e2", fact_id="f1", source="src")
    kg.index.build()
    return kg


def test_search_with_links_and_fact_confidence():
    kg = build_simple_graph()
    if kg.index._vectorizer is None:
        pytest.skip("sklearn missing")
    res = kg.search_with_links("hello", k=1, hops=1)
    assert "c1" in res
    data = kg.search_with_links_data("hello", k=1, hops=1)
    assert any(item["id"] == "c2" for item in data)
    score = kg.fact_confidence("e1", "likes", "e2")
    assert score >= 1.0
