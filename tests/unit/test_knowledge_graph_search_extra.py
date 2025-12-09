import types

from datacreek.core.knowledge_graph import KnowledgeGraph


def build_graph():
    kg = KnowledgeGraph()
    kg.add_document("doc1", "src1", text="Hello world")
    kg.add_chunk("doc1", "c1", "foo bar")
    kg.add_chunk("doc1", "c2", "hello baz")
    return kg


def test_search_documents_and_chunks():
    kg = build_graph()
    assert kg.search("doc1", node_type="document") == ["doc1"]
    assert kg.search_chunks("hello") == ["c2"]


def test_search_embeddings_and_hybrid(monkeypatch):
    kg = build_graph()
    # stub embedding index
    monkeypatch.setattr(kg.index, "search", lambda q, k=5: [1, 0])
    monkeypatch.setattr(kg.index, "get_id", lambda idx: ["c1", "c2"][idx])
    res = kg.search_embeddings("hello", k=1, fetch_neighbors=False)
    assert res == ["c2"]
    monkeypatch.setattr(kg, "search", lambda q, node_type="chunk": ["c2"])
    monkeypatch.setattr(
        kg,
        "search_embeddings",
        lambda q, k=5, fetch_neighbors=False, node_type="chunk": ["c1"],
    )
    res = kg.search_hybrid("hello", k=2)
    assert res == ["c2", "c1"]


def test_remove_document_and_chunk(monkeypatch):
    kg = build_graph()
    dummy_index = types.SimpleNamespace(
        ids=["c1", "c2"], remove=lambda x: dummy_index.ids.remove(x), build=lambda: None
    )
    kg.index = dummy_index
    kg.remove_chunk("c1")
    assert "c1" not in kg.graph
    assert "c1" not in dummy_index.ids
    kg.remove_document("doc1")
    assert "doc1" not in kg.graph
    assert not dummy_index.ids
