import pytest

from datacreek.core.knowledge_graph import KnowledgeGraph


def build_kg():
    kg = KnowledgeGraph()
    kg.add_document("d", "src")
    kg.add_chunk("d", "c1", "hello world")
    kg.add_chunk("d", "c2", "world hello")
    kg.add_chunk("d", "c3", "another text")
    return kg


def test_get_similar_chunks_data(monkeypatch):
    kg = build_kg()
    monkeypatch.setattr(
        kg.index,
        "nearest_neighbors",
        lambda k, return_distances=True: {"c1": [("c2", 0.9), ("c3", 0.8)]},
    )
    data = kg.get_similar_chunks_data("c1", k=2)
    assert data == [
        {"id": "c2", "similarity": 0.9, "text": "world hello", "document": "d"},
        {"id": "c3", "similarity": 0.8, "text": "another text", "document": "d"},
    ]


def test_get_chunk_neighbors(monkeypatch):
    kg = build_kg()
    monkeypatch.setattr(
        kg.index,
        "nearest_neighbors",
        lambda k: {"c1": ["c2", "c3"], "c2": ["c1"], "c3": ["c2", "c1"]},
    )
    neighbors = kg.get_chunk_neighbors(k=2)
    assert neighbors == {"c1": ["c2", "c3"], "c2": ["c1"], "c3": ["c2", "c1"]}
