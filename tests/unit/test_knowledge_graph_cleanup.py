import numpy as np

from datacreek.core.knowledge_graph import KnowledgeGraph


def test_deduplicate_and_prune(monkeypatch):
    kg = KnowledgeGraph()
    kg.add_document("d1", "s1")
    kg.add_chunk("d1", "c1", "Hello", source="s1")
    kg.add_chunk("d1", "c2", "Hello", source="s1")
    kg.index.build()
    assert kg.deduplicate_chunks() == 1
    assert "c2" not in kg.graph

    kg.add_document("d2", "s2")
    kg.add_chunk("d2", "c3", "text", source="s2")
    kg.graph.add_edge("c1", "c3", relation="r", provenance="s2")
    removed = kg.prune_sources(["s2"])
    assert removed == 2
    assert "d2" not in kg.graph and "c3" not in kg.graph
    assert not kg.graph.has_edge("c1", "c3")


def test_resolve_entities(monkeypatch):
    kg = KnowledgeGraph()
    kg.add_entity("e1", "Foo")
    kg.add_entity("e2", "foo")
    # ensure similarity calculation works without sklearn
    monkeypatch.setattr(
        "datacreek.core.knowledge_graph.cosine_similarity",
        lambda a, b: np.array([[1.0]]),
        raising=False,
    )
    monkeypatch.setattr(
        "datacreek.core.knowledge_graph.detect_language",
        lambda txt, return_prob=True: ("en", 1.0),
        raising=False,
    )
    monkeypatch.setattr(
        "datacreek.core.knowledge_graph.load_config",
        lambda: {"language": {"min_confidence": 0.0}},
    )
    monkeypatch.setattr(kg.index, "transform", lambda texts: np.zeros((len(texts), 2)))
    merged = kg.resolve_entities(threshold=0.0)
    assert merged == 1
    assert "e2" not in kg.graph


def test_deduplicate_chunks_fuzzy():
    kg = KnowledgeGraph()
    kg.add_document("d", "s")
    kg.add_chunk("d", "c1", "Hello world")
    kg.add_chunk("d", "c2", "Hello World!  ")
    kg.index.build()
    # fuzzy similarity <1 triggers SequenceMatcher path
    removed = kg.deduplicate_chunks(similarity=0.9)
    assert removed == 1
    assert ("c1" in kg.graph) != ("c2" in kg.graph)
