import numpy as np
import pytest

from datacreek.core.knowledge_graph import KnowledgeGraph


def build_meta_kg():
    kg = KnowledgeGraph()
    kg.add_document("d", "src")
    kg.add_chunk("d", "c1", "one")
    kg.add_chunk("d", "c2", "two")
    kg.graph.nodes["c1"]["embedding"] = [1.0, 0.0]
    kg.graph.nodes["c2"]["embedding"] = [0.0, 1.0]
    kg.graph.nodes["c1"]["graphwave_embedding"] = [0.1, 0.2]
    kg.graph.nodes["c2"]["graphwave_embedding"] = [0.2, 0.1]
    kg.graph.nodes["c1"]["poincare_embedding"] = [0.1, 0.3]
    kg.graph.nodes["c2"]["poincare_embedding"] = [0.2, 0.4]
    kg.graph.nodes["c1"]["fractal_level"] = 1
    kg.graph.nodes["c2"]["fractal_level"] = 1
    return kg


def test_aligned_cca_and_meta(monkeypatch, tmp_path):
    kg = build_meta_kg()
    # force cca_align path
    monkeypatch.setattr("os.path.exists", lambda p: False)
    monkeypatch.setattr(
        "datacreek.analysis.multiview.cca_align",
        lambda n2v, gw, n_components, path: {k: np.ones(n_components) for k in n2v},
    )
    kg.compute_aligned_cca_embeddings(n_components=2, path=tmp_path / "cca.pkl")
    assert "acca_embedding" in kg.graph.nodes["c1"]
    # meta autoencoder stub
    monkeypatch.setattr(
        "datacreek.analysis.multiview.meta_autoencoder",
        lambda n2v, gw, hyp, bottleneck: ({n: np.zeros(1) for n in n2v}, {}),
    )
    with pytest.raises(NameError):
        kg.compute_meta_embeddings(bottleneck=1)


def test_fractalnet_helpers(monkeypatch):
    kg = build_meta_kg()
    monkeypatch.setattr(
        "datacreek.analysis.fractal.fractalnet_compress",
        lambda emb, levels: {1: np.array([1.0, 1.0])},
    )
    assert 1 in kg.fractalnet_compress()
    monkeypatch.setattr(
        "datacreek.analysis.compression.prune_fractalnet",
        lambda w, ratio: np.zeros_like(np.asarray(w)),
    )
    out = kg.prune_fractalnet_weights([1.0, 2.0])
    assert np.all(out == 0)
    monkeypatch.setattr(
        "datacreek.analysis.fractal.fractal_net_prune",
        lambda feats, tol: (None, {n: i for i, n in enumerate(feats)}),
    )
    mapping = kg.prune_embeddings(tol=0.5)
    assert mapping == {"c1": 0, "c2": 1}
    assert kg.graph.nodes["c1"]["pruned_class"] == 0
