import pytest

from datacreek.core.knowledge_graph import KnowledgeGraph


def build_graph():
    kg = KnowledgeGraph()
    kg.add_document("d1", "src")
    kg.add_section("d1", "s1")
    kg.add_chunk("d1", "c1", "t1", section_id="s1")
    kg.add_chunk("d1", "c2", "t2", section_id="s1")
    kg.add_chunk("d1", "c3", "t3", section_id="s1")
    kg.graph.add_edge("c1", "c2", relation="next_chunk")
    kg.graph.add_edge("c2", "c3", relation="next_chunk")
    kg.graph.add_edge("c2", "c1", relation="prev_chunk")
    kg.graph.add_edge("c3", "c2", relation="prev_chunk")
    return kg


def test_get_chunk_context_full_range():
    kg = build_graph()
    ctx = kg.get_chunk_context("c2", before=1, after=1)
    assert ctx == ["c1", "c2", "c3"]
    # ensure requesting wider window stops at boundaries
    ctx2 = kg.get_chunk_context("c1", before=1, after=2)
    assert ctx2 == ["c1", "c2", "c3"]


def test_get_document_for_chunk_missing():
    kg = KnowledgeGraph()
    kg.add_document("d1", "src")
    kg.add_chunk("d1", "c1", "t")
    assert kg.get_document_for_chunk("c1") == "d1"
    with pytest.raises(Exception):
        kg.get_document_for_chunk("unknown")
