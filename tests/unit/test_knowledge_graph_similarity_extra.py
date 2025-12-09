from datacreek.core.knowledge_graph import KnowledgeGraph


def build_kg():
    kg = KnowledgeGraph()
    kg.add_document("d1", "src", text="A")
    kg.add_document("d2", "src", text="B")
    kg.add_section("d1", "s1", title="T1")
    kg.add_section("d2", "s2", title="T2")
    kg.graph.add_node("junk", type="junk")
    return kg


def test_get_similar_sections_filters_types(monkeypatch):
    kg = build_kg()

    def fake_search(query, k=3):
        return ["junk", "s2"]

    monkeypatch.setattr(kg.index, "search", fake_search)
    monkeypatch.setattr(kg.index, "get_id", lambda i: i)
    assert kg.get_similar_sections("s1", k=2) == ["s2"]


def test_get_similar_documents_filters_types(monkeypatch):
    kg = build_kg()

    def fake_search(query, k=3):
        return ["junk", "d2"]

    monkeypatch.setattr(kg.index, "search", fake_search)
    monkeypatch.setattr(kg.index, "get_id", lambda i: i)
    assert kg.get_similar_documents("d1", k=2) == ["d2"]
