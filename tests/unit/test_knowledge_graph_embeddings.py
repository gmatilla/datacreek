import sys
import types

import numpy as np

from datacreek.core.knowledge_graph import KnowledgeGraph


def test_embedding_algorithms(monkeypatch):
    kg = KnowledgeGraph()
    kg.add_document("d", "src")
    kg.add_chunk("d", "c1", "hello")

    class DummyWV(dict):
        def __getitem__(self, key):
            return np.array([0.0, 0.0])

    class DummyModel:
        wv = DummyWV()

    class DummyNode2Vec:
        def __init__(self, *a, **k):
            pass

        def fit(self):
            return DummyModel()

    # Stub node2vec to avoid heavy dependency requirement
    monkeypatch.setitem(
        sys.modules, "node2vec", types.SimpleNamespace(Node2Vec=DummyNode2Vec)
    )

    kg.compute_node2vec_embeddings(dimensions=2, walk_length=1, num_walks=1)
    assert "embedding" in kg.graph.nodes["d"]

    monkeypatch.setattr(
        "datacreek.analysis.fractal.graphwave_embedding",
        lambda g, s, num_points: {n: np.array([1.0, 0.0]) for n in g},
    )
    kg.compute_graphwave_embeddings([1.0], num_points=2)
    assert "graphwave_embedding" in kg.graph.nodes["d"]

    monkeypatch.setattr(
        "datacreek.analysis.fractal.poincare_embedding",
        lambda g, **k: {n: np.array([0.5, 0.5]) for n in g},
    )
    kg.compute_poincare_embeddings(
        dim=2, negative=1, epochs=1, learning_rate=0.1, burn_in=1
    )
    assert "poincare_embedding" in kg.graph.nodes["d"]


def test_relational_and_product_embeddings(monkeypatch):
    """Validate relational embeddings and product manifold helper."""
    kg = KnowledgeGraph()
    kg.add_document("d1", "src")
    kg.add_document("d2", "src")
    kg.graph.add_edge("d1", "d2", relation="related")
    kg.graph.nodes["d1"]["embedding"] = [0.1, 0.0]
    kg.graph.nodes["d2"]["embedding"] = [0.0, 0.1]

    kg.compute_transe_embeddings(dimensions=2)
    assert "transe_embedding" in kg.graph.edges["d1", "d2"]

    kg.compute_distmult_embeddings(dimensions=2)
    assert "distmult_embedding" in kg.graph.edges["d1", "d2"]

    monkeypatch.setattr(
        KnowledgeGraph,
        "compute_node2vec_embeddings",
        lambda self, **k: [
            self.graph.nodes[n].update({"embedding": [0.2, 0.2]})
            for n in self.graph.nodes
        ],
    )
    monkeypatch.setattr(
        KnowledgeGraph,
        "compute_graphwave_embeddings",
        lambda self, scales, num_points: [
            self.graph.nodes[n].update({"graphwave_embedding": [0.3, 0.3]})
            for n in self.graph.nodes
        ],
    )
    monkeypatch.setattr(
        KnowledgeGraph,
        "compute_poincare_embeddings",
        lambda self, **k: [
            self.graph.nodes[n].update({"poincare_embedding": [0.4, 0.4]})
            for n in self.graph.nodes
        ],
    )

    kg.compute_multigeometric_embeddings(
        node2vec_dim=2,
        graphwave_scales=[1.0],
        graphwave_points=2,
        poincare_dim=2,
        negative=1,
        epochs=1,
        burn_in=1,
    )
    assert kg.graph.nodes["d1"]["poincare_embedding"] == [0.4, 0.4]

    def fake_prod(h, e):
        return {n: np.concatenate([h[n], e[n]]) for n in h}

    # patch the analysis entrypoint to avoid heavy multiview dependency
    import datacreek.analysis.multiview as mv

    monkeypatch.setattr(mv, "product_embedding", fake_prod)
    kg.compute_product_manifold_embeddings(write_property="prod")
    assert np.allclose(kg.graph.nodes["d1"]["prod"], [0.4, 0.4, 0.2, 0.2])


def test_graphsage_and_hyper_embeddings(monkeypatch):
    kg = KnowledgeGraph()
    kg.add_document("d", "src")
    kg.add_chunk("d", "c1", "t1")
    kg.add_chunk("d", "c2", "t2")
    # ensure embeddings exist for GraphSAGE
    monkeypatch.setattr(
        KnowledgeGraph,
        "compute_node2vec_embeddings",
        lambda self, **k: [
            self.graph.nodes[n].update({"embedding": [1.0, 0.0]})
            for n in self.graph.nodes
        ],
    )

    class DummyPCA:
        def __init__(self, n_components, random_state=0):
            self.n_components = n_components

        def fit_transform(self, X):
            return X[:, : self.n_components]

    monkeypatch.setattr("datacreek.core.knowledge_graph.PCA", DummyPCA)
    kg.compute_graphsage_embeddings(dimensions=2, num_layers=1)
    assert "graphsage_embedding" in kg.graph.nodes["d"]

    # prepare hyperedge and node embeddings
    kg.graph.add_node("he", type="hyperedge")
    kg.graph.add_edge("he", "c1")
    kg.graph.add_edge("he", "c2")
    for n in ["c1", "c2"]:
        kg.graph.nodes[n]["embedding"] = [0.5, 0.5]

    monkeypatch.setattr(
        "datacreek.analysis.hypergraph.hyper_sagnn_embeddings",
        lambda edges, feats, embed_dim=None, seed=None: np.array([[0.1, 0.2]]),
    )
    out = kg.compute_hyper_sagnn_embeddings(embed_dim=2)
    assert out == {"he": [0.1, 0.2]}
    assert kg.graph.nodes["he"]["hyper_sagnn_embedding"] == [0.1, 0.2]

    monkeypatch.setattr(
        "datacreek.analysis.hypergraph.hyper_sagnn_head_drop_embeddings",
        lambda edges, feats, num_heads=4, threshold=0.1, seed=None: np.array([[1.0]]),
    )
    out_hd = kg.compute_hyper_sagnn_head_drop_embeddings()
    assert out_hd == {"he": [1.0]}
    assert kg.graph.nodes["he"]["hyper_sagnn_hd_embedding"] == [1.0]


def test_train_product_manifold_and_hyper_aa(monkeypatch):
    kg = KnowledgeGraph()
    kg.add_document("d1", "src")
    kg.add_document("d2", "src")
    kg.graph.nodes["d1"]["poincare_embedding"] = [0.1]
    kg.graph.nodes["d1"]["embedding"] = [0.2]
    kg.graph.nodes["d2"]["poincare_embedding"] = [0.3]
    kg.graph.nodes["d2"]["embedding"] = [0.4]
    monkeypatch.setattr(
        "datacreek.analysis.multiview.train_product_manifold",
        lambda h, e, ctx, alpha=0.5, lr=0.01, epochs=1: (
            {k: np.array([0.5]) for k in h},
            {k: np.array([0.6]) for k in e},
        ),
    )
    kg.train_product_manifold_embeddings([("d1", "d2")], epochs=1)
    assert kg.graph.nodes["d1"]["poincare_embedding"] == [0.5]
    assert kg.graph.nodes["d1"]["embedding"] == [0.6]

    kg.graph.add_node("he1", type="hyperedge")
    kg.graph.add_edge("he1", "d1")
    kg.graph.add_edge("he1", "d2")
    monkeypatch.setattr(
        "datacreek.analysis.hypergraph.hyper_adamic_adar_scores",
        lambda edges: {("d1", "d2"): 0.7},
    )
    scores = kg.hyper_adamic_adar_scores()
    assert scores == {("d1", "d2"): 0.7}
