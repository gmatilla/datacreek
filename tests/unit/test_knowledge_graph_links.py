import pytest

from datacreek.core.knowledge_graph import KnowledgeGraph


def build_graph():
    kg = KnowledgeGraph()
    kg.add_document("d1", "src", author="auth", organization="org")
    kg.add_section("d1", "s1", title="T1")
    kg.add_chunk("d1", "c1", "one", section_id="s1")
    kg.add_chunk("d1", "c2", "two", section_id="s1")
    kg.add_document("d2", "src2", author="auth", organization="org")
    kg.add_section("d2", "s2", title="T2")
    kg.add_chunk("d2", "c3", "three", section_id="s2")
    for cid in ["c1", "c2", "c3"]:
        kg.add_entity(f"e{cid}", cid)
        kg.link_entity(cid, f"e{cid}")
    kg.index.build()
    return kg


def test_link_helpers_and_conflicts():
    kg = build_graph()
    assert kg.link_chunks_by_entity() >= 0
    assert kg.link_documents_by_entity() >= 0
    assert kg.link_sections_by_entity() >= 0
    assert kg.link_authors_organizations() >= 1
    count = kg.mark_conflicting_facts()
    assert isinstance(count, int)


def test_remove_and_renumber():
    kg = build_graph()
    kg.remove_chunk("c1")
    assert kg.get_chunks_for_document("d1") == ["c2"]
    kg.remove_document("d2")
    assert "d2" not in kg.graph
    kg.consolidate_schema()
    assert kg.graph.nodes["d1"]["type"] == "document"
