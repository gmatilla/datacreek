import pytest

from datacreek.core.knowledge_graph import KnowledgeGraph


def build_graph():
    kg = KnowledgeGraph()
    kg.add_document("d1", "src1")
    kg.add_section("d1", "s1", title="A", page=1)
    kg.add_chunk("d1", "c1", "hello", section_id="s1", page=1)
    kg.add_document("d2", "src2")
    kg.add_chunk("d2", "c2", "world", page=2)
    kg.add_entity("e1", "Alice")
    kg.add_entity("e2", "Bob")
    kg.add_fact("e1", "knows", "e2", fact_id="f1")
    kg.link_entity("c1", "e1")
    kg.graph.add_edge("c1", "f1", relation="has_fact")
    kg.index.build()
    return kg


def test_fact_and_entity_queries():
    kg = build_graph()
    # queries through fact relationships
    assert kg.get_facts_for_entity("e1") == ["f1"]
    assert kg.get_entities_for_fact("f1") == ["e1", "e2"]
    assert kg.get_chunks_for_fact("f1") == ["c1"]
    assert kg.get_sections_for_fact("f1") == ["s1"]
    assert kg.get_documents_for_fact("f1") == ["d1"]
    assert kg.get_pages_for_fact("f1") == [1]
    # entity based queries
    assert kg.get_chunks_for_entity("e1") == ["c1"]
    assert kg.get_documents_for_entity("e1") == ["d1"]
    assert kg.get_pages_for_entity("e1") == [1]
    # fact lookup helpers
    assert kg.find_facts(predicate="knows") == ["f1"]


def test_chunk_context_helpers():
    kg = build_graph()
    kg.graph.add_edge("c1", "c2", relation="next_chunk")
    context = kg.get_chunk_context("c1", before=0, after=1)
    assert context == ["c1", "c2"]
