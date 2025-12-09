import types

import numpy as np

from datacreek.core.knowledge_graph import KnowledgeGraph


class DummySession:
    def __init__(self):
        self.queries = []

    def run(self, query, **params):
        self.queries.append(query)

        class DummyResult(list):
            def single(self):
                return self[0] if self else None

        if "wcc.stream" in query:
            return DummyResult([{"nodeId": 1, "componentId": 0}])
        if (
            "nodeSimilarity.stream('kg_qc')" in query
            or "nodeSimilarity.stream('kg_sim')" in query
        ):
            return DummyResult([{"node1": 1, "node2": 2, "similarity": 0.9}])
        if "adamicAdar.stream" in query:
            return DummyResult([{"sourceNodeId": 1, "targetNodeId": 2, "score": 0.8}])
        if "preferentialAttachment.stream" in query:
            return []
        if "degree.stream" in query or "betweenness.stream" in query:
            return DummyResult([])
        if "triangleCount.stream" in query:
            return DummyResult([])
        if "MATCH (n {id:$node_id" in query:
            return DummyResult([{"nid": 1}])
        if "MATCH (n) WHERE id(n)=$id RETURN n.id AS name" in query:
            return DummyResult([{"name": "node2"}])
        return DummyResult([])

    def execute_read(self, fn):
        return fn(self)

    def session(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class DummyDriver:
    def __init__(self):
        self.session_obj = DummySession()

    def session(self):
        return self.session_obj


def test_gds_quality_and_node_similarity(monkeypatch):
    kg = KnowledgeGraph()
    kg.add_document("d1", "src")
    kg.add_chunk("d1", "c1", "text")
    kg.add_chunk("d1", "c2", "more")
    kg.index.build()
    driver = DummyDriver()
    monkeypatch.setattr(
        "datacreek.core.knowledge_graph.get_cleanup_cfg",
        lambda: {
            "k_min": 1,
            "sigma": 0.5,
            "tau": 1,
            "lp_sigma": 0.0,
            "lp_topk": 1,
            "hub_deg": 1,
        },
    )
    res = kg.gds_quality_check(driver, link_threshold=0.7)
    assert res["removed_nodes"] == []
    assert res["duplicates"] == [(1, 2, 0.9)]
    assert ("c1", "c2") in kg.graph.edges or ("c2", "c1") in kg.graph.edges

    driver2 = DummyDriver()
    matches = kg.node_similarity(driver2, "c1", threshold=0.8)
    assert matches == [("node2", 0.9)]


def test_mark_conflicts():
    kg = KnowledgeGraph()
    kg.graph.add_node("a", type="entity")
    kg.graph.add_node("b", type="entity")
    kg.graph.add_node("c", type="entity")
    kg.graph.add_edge("a", "b", relation="likes")
    kg.graph.add_edge("a", "c", relation="likes")
    count = kg.mark_conflicting_facts()
    assert count == 2
    assert kg.graph.edges["a", "b"]["conflict"] is True
    assert kg.graph.edges["a", "c"]["conflict"] is True
