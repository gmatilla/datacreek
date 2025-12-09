import pytest

from datacreek.core.knowledge_graph import KnowledgeGraph


def build_graph():
    kg = KnowledgeGraph()
    kg.add_document("doc", "src")
    kg.add_section("doc", "sec1", title="First")
    kg.add_section("doc", "sec2", title="Second")
    kg.add_chunk("doc", "c1", "one", section_id="sec1")
    kg.add_chunk("doc", "c2", "two", section_id="sec1")
    kg.add_chunk("doc", "c3", "three", section_id="sec2")
    kg.add_audio("doc", "a1", "/a", lang="en")
    kg.link_transcript("c3", "a1")
    return kg


def test_section_and_chunk_navigation():
    kg = build_graph()
    # section navigation
    assert kg.get_next_section("sec1") == "sec2"
    assert kg.get_previous_section("sec2") == "sec1"
    # chunk navigation
    assert kg.get_next_chunk("c1") == "c2"
    assert kg.get_previous_chunk("c2") == "c1"
    # chunk to section mapping
    assert kg.get_section_for_chunk("c1") == "sec1"
    assert kg.get_section_for_chunk("c3") == "sec2"


def test_transcript_link():
    kg = build_graph()
    edges = list(kg.graph.edges("c3", data=True))
    assert any(e[1] == "a1" and e[2].get("relation") == "transcript_of" for e in edges)
