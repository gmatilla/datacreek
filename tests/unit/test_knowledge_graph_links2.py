from datacreek.core.knowledge_graph import KnowledgeGraph


def build_graph():
    kg = KnowledgeGraph()
    kg.add_document("d1", "src", author="a1", organization="o1")
    kg.add_document("d2", "src", author="a1", organization="o1")
    kg.add_entity("a1", "Author")
    kg.add_entity("o1", "Org")
    kg.link_authors_organizations()
    kg.add_section("d1", "s1")
    kg.add_section("d2", "s2")
    kg.add_chunk("d1", "c1", "t1", section_id="s1")
    kg.add_chunk("d2", "c2", "t2", section_id="s2")
    kg.add_entity("e1", "E1")
    kg.link_entity("c1", "e1", provenance="src")
    kg.link_entity("c2", "e1")
    return kg


def test_entity_link_helpers_and_serialization():
    kg = build_graph()
    assert kg.link_documents_by_entity() == 1
    assert kg.link_sections_by_entity() == 1
    d = kg.to_dict()
    rebuilt = KnowledgeGraph.from_dict(d)
    assert set(rebuilt.graph.nodes) == set(kg.graph.nodes)
    assert set(rebuilt.graph.edges) >= set(kg.graph.edges)
    text = rebuilt.to_text()
    assert "t1" in text and "t2" in text
