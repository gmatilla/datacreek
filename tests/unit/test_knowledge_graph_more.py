import numpy as np

from datacreek.core.knowledge_graph import KnowledgeGraph


def test_embeddings_and_clustering(monkeypatch):
    kg = KnowledgeGraph()
    kg.add_document("d1", "src")
    kg.add_chunk("d1", "c1", "hello world")
    kg.add_chunk("d1", "c2", "hello there")
    kg.add_entity("e1", "foo")
    kg.link_entity("c1", "e1")

    # ensure deterministic embeddings
    monkeypatch.setattr(kg.index, "embed", lambda text: np.array([1.0, 0.0]))
    kg.update_embeddings()
    assert "embedding" in kg.graph.nodes["c1"]

    class DummyKMeans:
        def __init__(self, n_clusters, n_init):
            pass

        def fit_predict(self, X):
            return [0] * len(X)

    monkeypatch.setattr("datacreek.core.knowledge_graph.KMeans", DummyKMeans)

    kg.cluster_chunks(n_clusters=1)
    kg.cluster_entities(n_clusters=1)
    kg.summarize_communities()
    kg.summarize_entity_groups()
    kg.score_trust()

    assert any(d.get("type") == "community" for d in kg.graph.nodes.values())
    assert any(d.get("type") == "entity_group" for d in kg.graph.nodes.values())
    assert "summary" in kg.graph.nodes["community_0"]
    assert kg.graph.nodes["d1"].get("trust") == 1.0


def test_schema_and_entropy(monkeypatch):
    kg = KnowledgeGraph()
    kg.add_document("DOC", "src")
    kg.add_chunk("DOC", "CHUNK", "text")
    kg.graph.nodes["DOC"]["type"] = "Document"
    kg.graph.add_edge("DOC", "CHUNK", relation="HAS_CHUNK")
    kg.consolidate_schema()
    assert kg.graph.nodes["DOC"]["type"] == "document"
    assert kg.graph.edges["DOC", "CHUNK"]["relation"] == "has_chunk"

    kg.graph.nodes["DOC"]["embedding"] = [0.1, 0.2]
    kg.graph.nodes["CHUNK"]["embedding"] = [0.2, 0.3]
    monkeypatch.setattr(
        "datacreek.analysis.fractal.embedding_entropy", lambda emb: 0.123
    )
    assert kg.embedding_entropy() == 0.123

    monkeypatch.setattr(
        "datacreek.core.knowledge_graph.KnowledgeGraph.graphwave_entropy",
        lambda self: 0.05,
    )

    def fake_compute(self, scales, num_points):
        for n in self.graph.nodes:
            self.graph.nodes[n]["graphwave_embedding"] = [0.1, 0.1]

    monkeypatch.setattr(
        "datacreek.core.knowledge_graph.KnowledgeGraph.compute_graphwave_embeddings",
        fake_compute,
    )
    monkeypatch.setattr("numpy.random.uniform", lambda a, b: 0.0)
    assert kg.ensure_graphwave_entropy(0.1) == 0.05
