import types

import numpy as np

from datacreek.core.knowledge_graph import KnowledgeGraph


class DummyRequests:
    def __init__(self, data):
        self._data = data

    def get(self, *a, **k):
        return types.SimpleNamespace(json=lambda: self._data)


def test_enrich_entity_wikidata(monkeypatch):
    kg = KnowledgeGraph()
    kg.add_entity("e1", "Foo", source="s")
    dummy = DummyRequests({"search": [{"id": "Q1", "description": "desc"}]})
    monkeypatch.setattr("datacreek.core.knowledge_graph.requests", dummy)
    kg.enrich_entity_wikidata("e1")
    node = kg.graph.nodes["e1"]
    assert node["wikidata_id"] == "Q1"
    assert node["description"] == "desc"


def test_enrich_entity_dbpedia(monkeypatch):
    kg = KnowledgeGraph()
    kg.add_entity("e1", "Bar", source="s")
    dummy = DummyRequests({"docs": [{"uri": "u1", "description": "d"}]})
    monkeypatch.setattr("datacreek.core.knowledge_graph.requests", dummy)
    kg.enrich_entity_dbpedia("e1")
    node = kg.graph.nodes["e1"]
    assert node["dbpedia_uri"] == "u1"
    assert node["description_dbpedia"] == "d"


def test_predict_links(monkeypatch):
    kg = KnowledgeGraph()
    kg.add_entity("e1", "foo")
    kg.add_entity("e2", "bar")
    monkeypatch.setattr(
        "datacreek.core.knowledge_graph.cosine_similarity",
        lambda a, b: np.array([[1.0]]),
    )
    monkeypatch.setattr(
        kg.index, "transform", lambda t: np.array([[1.0, 0.0], [1.0, 0.0]])
    )
    kg.predict_links(threshold=0.5)
    assert kg.graph.has_edge("e1", "e2")
