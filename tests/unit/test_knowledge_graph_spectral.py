import pytest

from datacreek.core.knowledge_graph import KnowledgeGraph


def build_kg():
    kg = KnowledgeGraph()
    kg.add_document("d1", "s")
    kg.add_chunk("d1", "c1", "a")
    kg.add_chunk("d1", "c2", "b")
    kg.graph.add_edge("c1", "c2")
    return kg


def test_spectral_metrics():
    kg = build_kg()
    assert kg.spectral_entropy() >= 0
    assert kg.spectral_gap() >= 0
    assert kg.laplacian_energy() >= 0
    assert kg.lacunarity(radius=1) >= 0
