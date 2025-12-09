import pytest

from datacreek.core.knowledge_graph import KnowledgeGraph


def build_basic_graph():
    kg = KnowledgeGraph()
    kg.add_document("d", "src")
    kg.add_section("d", "s", title="t")
    kg.add_chunk("d", "c", "txt", section_id="s")
    kg.add_entity("e", "e text")
    kg.link_entity("c", "e")
    return kg


def test_duplicate_nodes_and_links():
    kg = build_basic_graph()
    with pytest.raises(ValueError):
        kg.add_document("d", "src")
    with pytest.raises(ValueError):
        kg.add_entity("e", "e text")
    kg.add_fact("e", "r", "e", fact_id="f")
    with pytest.raises(ValueError):
        kg.add_fact("e", "r", "e", fact_id="f")
    kg.add_section("d", "s2")
    with pytest.raises(ValueError):
        kg.add_section("d", "s2")
    kg.add_chunk("d", "c2", "foo", section_id="s")
    with pytest.raises(ValueError):
        kg.add_chunk("d", "c2", "foo", section_id="s")


def test_unknown_nodes_and_invalid_property():
    kg = build_basic_graph()
    with pytest.raises(ValueError):
        kg.link_entity("missing", "e")
    with pytest.raises(ValueError):
        kg.link_transcript("missing", "noaudio")
    kg.add_audio("d", "a1", "path", lang="en")
    with pytest.raises(ValueError):
        kg.add_audio("d", "a1", "path", lang="en")
    with pytest.raises(ValueError):
        kg.set_property("invalid space", 1)
