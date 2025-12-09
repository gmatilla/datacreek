import numpy as np
import pytest

from datacreek.analysis import (
    aligned_cca,
    cca_align,
    load_cca,
    meta_autoencoder,
    multiview_contrastive_loss,
    product_embedding,
    train_product_manifold,
)
from datacreek.core.dataset import DatasetBuilder, DatasetType


def test_product_embedding_simple():
    hyper = {"a": [0.0, 1.0], "b": [1.0, 0.0]}
    eucl = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    emb = product_embedding(hyper, eucl)
    assert len(emb["a"]) == 4
    assert np.allclose(emb["a"], [0.0, 1.0, 1.0, 0.0])


def test_aligned_cca_dimension():
    n2v = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    gw = {"a": [0.5, 0.5], "b": [-0.5, -0.5]}
    latent, _ = aligned_cca(n2v, gw, n_components=1)
    assert set(latent) == {"a", "b"}
    assert latent["a"].shape == (1,)


def test_dataset_wrappers_multiview():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "x")
    ds.add_chunk("d", "c2", "y")
    ds.add_entity("e1", "E")
    ds.add_entity("e2", "E")
    ds.link_entity("c1", "e1")
    ds.link_entity("c2", "e2")
    ds.graph.graph.nodes["e1"]["embedding"] = [1.0, 0.0]
    ds.graph.graph.nodes["e2"]["embedding"] = [0.0, 1.0]
    ds.graph.graph.nodes["e1"]["poincare_embedding"] = [0.0, 1.0]
    ds.graph.graph.nodes["e2"]["poincare_embedding"] = [1.0, 0.0]
    ds.graph.graph.nodes["e1"]["graphwave_embedding"] = [0.2, 0.2]
    ds.graph.graph.nodes["e2"]["graphwave_embedding"] = [-0.2, -0.2]
    ds.compute_product_manifold_embeddings()
    ds.compute_aligned_cca_embeddings(n_components=1)
    node = ds.graph.graph.nodes["e1"]
    assert "product_embedding" in node
    assert "acca_embedding" in node


def test_multiview_contrastive_and_meta():
    n2v = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    gw = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    hyp = {"a": [0.1, 0.0], "b": [0.0, 0.1]}
    loss = multiview_contrastive_loss(n2v, gw, hyp, tau=0.5)
    assert loss >= 0.0
    latent, recon = meta_autoencoder(n2v, gw, hyp, bottleneck=1)
    assert set(latent) == {"a", "b"}
    assert recon["a"].shape == (6,)


def test_train_product_manifold():
    hyper = {"a": [0.0, 0.5], "b": [0.5, 0.0]}
    eucl = {"a": [1.0, 0.0], "b": [0.0, 1.0]}

    def loss(h, e):
        dh = np.linalg.norm(h["a"] - h["b"]) ** 2
        de = np.linalg.norm(e["a"] - e["b"]) ** 2
        return dh + de

    before = loss(
        {k: np.array(v) for k, v in hyper.items()},
        {k: np.array(v) for k, v in eucl.items()},
    )
    h_new, e_new = train_product_manifold(hyper, eucl, [("a", "b")], epochs=5, lr=0.1)
    after = loss(h_new, e_new)
    assert after < before


def test_hybrid_score_and_similarity():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_entity("e1", "A")
    ds.add_entity("e2", "B")
    ds.graph.graph.nodes["e1"].update(
        {
            "embedding": [1.0, 0.0],
            "poincare_embedding": [0.0, 0.5],
            "graphwave_embedding": [0.1, 0.0],
        }
    )
    ds.graph.graph.nodes["e2"].update(
        {
            "embedding": [0.9, 0.1],
            "poincare_embedding": [0.0, 0.6],
            "graphwave_embedding": [0.1, 0.1],
        }
    )
    score = ds.hybrid_score("e1", "e2")
    assert score != 0.0
    sims = ds.similar_by_hybrid("e1", node_type="entity", k=1)
    assert sims[0][0] == "e2"


def test_ann_hybrid_search():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello world")
    ds.add_chunk("d", "c2", "hola mundo")
    ds.graph.graph.nodes["c1"].update(
        {
            "embedding": [1.0, 0.0],
            "graphwave_embedding": [0.5, 0.5],
            "poincare_embedding": [0.0, 1.0],
        }
    )
    ds.graph.graph.nodes["c2"].update(
        {
            "embedding": [0.9, 0.1],
            "graphwave_embedding": [0.4, 0.6],
            "poincare_embedding": [0.1, 0.9],
        }
    )
    ds.build_faiss_index("embedding")
    results = ds.ann_hybrid_search([1.0, 0.0], [0.5, 0.5], [0.0, 1.0])
    assert results[0][0] in {"c1", "c2"}


def test_dataset_train_product_manifold():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_entity("e1", "A")
    ds.add_entity("e2", "B")
    ds.graph.graph.nodes["e1"].update(
        {"embedding": [1.0, 0.0], "poincare_embedding": [0.0, 0.5]}
    )
    ds.graph.graph.nodes["e2"].update(
        {"embedding": [0.0, 1.0], "poincare_embedding": [0.5, 0.0]}
    )
    before = (
        np.linalg.norm(
            np.array(ds.graph.graph.nodes["e1"]["embedding"])
            - np.array(ds.graph.graph.nodes["e2"]["embedding"])
        )
        ** 2
    )
    ds.train_product_manifold_embeddings([("e1", "e2")], epochs=5, lr=0.1)
    after = (
        np.linalg.norm(
            np.array(ds.graph.graph.nodes["e1"]["embedding"])
            - np.array(ds.graph.graph.nodes["e2"]["embedding"])
        )
        ** 2
    )
    assert after < before


def test_cca_align_persists(tmp_path):
    n2v = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    gw = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    path = tmp_path / "cca.pkl"
    latent = cca_align(n2v, gw, n_components=1, path=str(path))
    assert path.exists()
    import pickle

    with open(path, "rb") as f:
        w = pickle.load(f)
    assert set(w) == {"Wn2v", "Wgw"}
    assert set(latent) == {"a", "b"}


def test_load_cca_roundtrip(tmp_path):
    n2v = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    gw = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    path = tmp_path / "cca.pkl"
    cca_align(n2v, gw, n_components=1, path=str(path))
    W1, W2 = load_cca(str(path))
    assert W1.shape[0] == 2
    assert W2.shape[0] == 2
    assert not W1.flags.writeable
    assert not W2.flags.writeable


def test_load_cca_logs_sha(tmp_path, caplog):
    n2v = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    gw = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    path = tmp_path / "cca.pkl"
    cca_align(n2v, gw, n_components=1, path=str(path))
    caplog.set_level("INFO")
    load_cca(str(path))
    assert any("cca_sha=" in r.message for r in caplog.records)


def test_cca_align_logs_sha(tmp_path, caplog):
    n2v = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    gw = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    path = tmp_path / "cca.pkl"
    caplog.set_level("INFO")
    cca_align(n2v, gw, n_components=1, path=str(path))
    assert any("cca_sha=" in r.message for r in caplog.records)


def test_load_cca_missing(tmp_path):
    missing = tmp_path / "none.pkl"
    with pytest.raises(FileNotFoundError):
        load_cca(str(missing))


def test_compute_aligned_cca_embeddings_cache(monkeypatch, tmp_path):
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_entity("e1", "E")
    ds.add_entity("e2", "E")
    for node in ["e1", "e2"]:
        ds.graph.graph.nodes[node]["embedding"] = (
            [1.0, 0.0] if node == "e1" else [0.0, 1.0]
        )
        ds.graph.graph.nodes[node]["graphwave_embedding"] = (
            [1.0, 0.0] if node == "e1" else [0.0, 1.0]
        )
    path = tmp_path / "cca.pkl"
    ds.compute_aligned_cca_embeddings(n_components=1, path=str(path))
    assert path.exists()

    def fail(*a, **k):
        raise AssertionError("aligned_cca should not run")

    monkeypatch.setattr("datacreek.analysis.multiview.aligned_cca", fail)
    ds.compute_aligned_cca_embeddings(n_components=1, path=str(path))
    assert "acca_embedding" in ds.graph.graph.nodes["e1"]
