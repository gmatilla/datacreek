import pytest

from datacreek.core.knowledge_graph import KnowledgeGraph


def build_graph():
    kg = KnowledgeGraph()
    kg.add_document("d", "src")
    # three chunks with text so index works
    kg.add_chunk("d", "c1", "hello world")
    kg.add_chunk("d", "c2", "world hello")
    kg.add_chunk("d", "c3", "another text")
    return kg


def test_get_chunk_neighbors_data(monkeypatch):
    kg = build_graph()
    # stub nearest_neighbors to avoid heavy embedding search
    monkeypatch.setattr(
        kg.index,
        "nearest_neighbors",
        lambda k, return_distances=True: {
            "c1": [("c2", 0.9), ("c3", 0.8)],
            "c2": [("c1", 0.9)],
            "c3": [("c2", 0.8)],
        },
    )
    data = kg.get_chunk_neighbors_data(k=2)
    assert data["c1"][0]["id"] == "c2"
    assert data["c1"][0]["similarity"] == 0.9
    assert data["c1"][0]["text"] == "world hello"
    assert data["c1"][0]["document"] == "d"
    assert data["c2"][0]["id"] == "c1"
    assert data["c2"][0]["similarity"] == 0.9
    assert data["c3"][0]["id"] == "c2"
    assert data["c3"][0]["similarity"] == 0.8
