import sys
import types

import numpy as np
import pytest

from datacreek.core.knowledge_graph import (
    CleanupConfig,
    KnowledgeGraph,
    apply_cleanup_config,
)


def build_graph():
    kg = KnowledgeGraph()
    kg.add_document("d1", "src", author="Author", organization="Org")
    kg.add_section("d1", "s1", title="Intro")
    kg.add_chunk("d1", "c1", "<b>Hello</b>", section_id="s1")
    kg.add_chunk("d1", "c2", "Hello", section_id="s1")
    kg.add_entity("e1", "Entity")
    kg.link_entity("c1", "e1")
    kg.link_entity("c2", "e1")
    kg.index.build()
    return kg


def test_misc_operations(monkeypatch):
    kg = build_graph()
    # stub BeautifulSoup
    bs4 = types.SimpleNamespace(
        BeautifulSoup=lambda txt, parser: types.SimpleNamespace(
            get_text=lambda sep: txt.replace("<b>", "").replace("</b>", "")
        )
    )
    monkeypatch.setitem(sys.modules, "bs4", bs4)
    # patch nearest_neighbors
    monkeypatch.setattr(
        kg.index,
        "nearest_neighbors",
        lambda k, return_distances=True: {"c1": [("c2", 0.9)]},
    )
    kg.link_similar_chunks()
    assert kg.graph.edges["c1", "c2"].get("relation") == "similar_to"
    assert kg.clean_chunk_texts() == 1
    kg.graph.nodes["c1"]["birth_date"] = "2025-01-01T00:00:00"
    assert kg.normalize_date_fields() == 1
    kg.graph.nodes["c2"]["birth_date"] = "2024-01-01"
    kg.graph.add_edge("c1", "c2", relation="parent_of")
    assert kg.validate_coherence() == 1
    kg.link_chunks_by_entity()
    kg.link_documents_by_entity()
    kg.link_sections_by_entity()
    kg.link_authors_organizations()
    kg.graph.nodes["c1"]["text"] = "Hello"
    kg.graph.nodes["c2"]["text"] = "Hello"
    assert kg.deduplicate_chunks() == 1
    kg.graph.add_node("d_extra", type="document", source="src")
    assert kg.prune_sources(["src"]) >= 1
    kg.graph.add_edge("c1", "c2", relation="knows")
    kg.mark_conflicting_facts()
    data = kg.to_dict()
    rebuilt = KnowledgeGraph.from_dict(data)
    assert set(rebuilt.graph.nodes) == set(kg.graph.nodes)
    text = rebuilt.to_text()
    assert isinstance(text, str)
    kg.set_property("foo", 1)
