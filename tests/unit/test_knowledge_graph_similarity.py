from datacreek.core.knowledge_graph import KnowledgeGraph


def test_link_similar_sections_and_documents(monkeypatch):
    kg = KnowledgeGraph()
    kg.add_document("d1", "src")
    kg.add_section("d1", "s1", title="A")
    kg.add_section("d1", "s2", title="B")
    kg.add_document("d2", "src")
    kg.add_section("d2", "s3", title="C")

    def fake_nn(k, return_distances=True):
        return {
            "s1": [("s2", 0.9)],
            "s3": [("s1", 0.8)],
            "d1": [("d2", 0.7)],
        }

    monkeypatch.setattr(kg.index, "nearest_neighbors", fake_nn)
    kg.link_similar_sections(k=1)
    kg.link_similar_documents(k=1)
    assert ("s1", "s2") in kg.graph.edges or ("s2", "s1") in kg.graph.edges
    assert ("d1", "d2") in kg.graph.edges or ("d2", "d1") in kg.graph.edges


def test_get_similar_sections_and_documents(monkeypatch):
    kg = KnowledgeGraph()
    kg.add_document("d1", "src", text="hello world")
    kg.add_section("d1", "s1", title="Intro")
    kg.add_document("d2", "src2", text="hello there")
    kg.add_section("d2", "s2", title="Intro2")

    def fake_search(query, k=3):
        if query == "Intro":
            return ["s1", "s2"]
        if query == "hello world":
            return ["d1", "d2"]
        return []

    monkeypatch.setattr(kg.index, "search", fake_search)
    monkeypatch.setattr(kg.index, "get_id", lambda idx: idx)

    assert kg.get_similar_sections("s1", k=1) == ["s2"]
    assert kg.get_similar_documents("d1", k=1) == ["d2"]
