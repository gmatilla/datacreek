import numpy as np

import datacreek.core.knowledge_graph as kgmod
from datacreek.core.knowledge_graph import KnowledgeGraph


def test_langid_prevents_cross_language_merge(monkeypatch):
    monkeypatch.setattr(
        "datacreek.core.knowledge_graph.cosine_similarity",
        lambda a, b: np.ones((1, 1)),
    )
    monkeypatch.setattr(
        "datacreek.utils.retrieval.EmbeddingIndex.transform",
        lambda self, t: np.zeros((len(t), 1)),
    )

    def detect(text, **k):
        if "Hello" in text:
            return ("en", 0.99) if k.get("return_prob") else "en"
        return ("fr", 0.99) if k.get("return_prob") else "fr"

    monkeypatch.setattr("datacreek.utils.text.detect_language", detect)
    kg = KnowledgeGraph()
    kg.add_entity("e1", "Hello world")
    kg.add_entity("e2", "Bonjour le monde")
    merged = kg.resolve_entities(threshold=0.0)
    assert merged == 0


def test_langid_allows_same_language(monkeypatch):
    monkeypatch.setattr(
        "datacreek.core.knowledge_graph.load_config",
        lambda: {"language": {"min_confidence": 0.7}},
    )

    def detect(text, **k):
        return ("en", 0.9) if k.get("return_prob") else "en"

    monkeypatch.setattr("datacreek.utils.text.detect_language", detect)
    monkeypatch.setattr(
        "datacreek.utils.retrieval.EmbeddingIndex.transform",
        lambda self, t: np.zeros((len(t), 1)),
    )
    monkeypatch.setattr(
        "datacreek.core.knowledge_graph.cosine_similarity",
        lambda a, b: np.ones((1, 1)),
    )
    kg = KnowledgeGraph()
    kg.add_entity("e1", "Hello world")
    kg.add_entity("e2", "Hi world")
    merged = kg.resolve_entities(threshold=0.0)
    assert merged == 1


def test_langid_prob_allows_merge(monkeypatch):
    monkeypatch.setattr(
        "datacreek.core.knowledge_graph.load_config",
        lambda: {"language": {"min_confidence": 0.7}},
    )

    def detect(text, **k):
        if "Hello" in text:
            return ("en", 0.9) if k.get("return_prob") else "en"
        return ("fr", 0.95) if k.get("return_prob") else "fr"

    monkeypatch.setattr("datacreek.utils.text.detect_language", detect)
    monkeypatch.setattr(
        "datacreek.utils.retrieval.EmbeddingIndex.transform",
        lambda self, t: np.zeros((len(t), 1)),
    )
    monkeypatch.setattr(
        "datacreek.core.knowledge_graph.cosine_similarity",
        lambda a, b: np.ones((1, 1)),
    )
    kg = KnowledgeGraph()
    kg.add_entity("e1", "Hello world")
    kg.add_entity("e2", "Bonjour le monde")
    merged = kg.resolve_entities(threshold=0.0)
    assert merged == 0


def test_langid_mismatch_counter(monkeypatch):
    monkeypatch.setattr(
        "datacreek.core.knowledge_graph.load_config",
        lambda: {"language": {"min_confidence": 0.8}},
    )
    calls = {"n": 0}

    class DummyCounter:
        def inc(self):
            calls["n"] += 1

    monkeypatch.setattr(
        "datacreek.analysis.monitoring.lang_mismatch_total", DummyCounter()
    )

    def detect(text, **k):
        if "Hello" in text:
            return ("en", 0.5) if k.get("return_prob") else "en"
        return ("fr", 0.5) if k.get("return_prob") else "fr"

    monkeypatch.setattr("datacreek.utils.text.detect_language", detect)
    monkeypatch.setattr(
        "datacreek.utils.retrieval.EmbeddingIndex.transform",
        lambda self, t: np.zeros((len(t), 1)),
    )
    monkeypatch.setattr(
        "datacreek.core.knowledge_graph.cosine_similarity",
        lambda a, b: np.ones((1, 1)),
    )
    kg = KnowledgeGraph()
    kg.add_entity("e1", "Hello world")
    kg.add_entity("e2", "Bonjour le monde")
    merged = kg.resolve_entities(threshold=0.0)
    assert merged == 0
    assert calls["n"] == 1
