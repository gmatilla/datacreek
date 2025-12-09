import sys
import types

import numpy as np
import pytest

from datacreek.analysis import multiview


class DummyCCA:
    def __init__(self, n_components=1):
        self.n_components = n_components
        self.x_weights_ = None
        self.y_weights_ = None

    def fit_transform(self, X, Y):
        self.x_weights_ = np.ones((X.shape[1], self.n_components))
        self.y_weights_ = np.ones((Y.shape[1], self.n_components))
        return X[:, : self.n_components], Y[:, : self.n_components]


class DummyPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.dim = None

    def fit_transform(self, X):
        self.dim = X.shape[1]
        return X[:, : self.n_components]

    def inverse_transform(self, Z):
        out = np.zeros((Z.shape[0], self.dim))
        out[:, : Z.shape[1]] = Z
        return out


@pytest.mark.heavy
def test_product_embedding_and_train():
    hyper = {"a": [0.0, 0.5], "b": [0.5, 0.0]}
    eucl = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    emb = multiview.product_embedding(hyper, eucl)
    assert np.allclose(emb["a"], [0.0, 0.5, 1.0, 0.0])
    before = np.linalg.norm(np.array(hyper["a"]) - np.array(hyper["b"])) ** 2
    h_new, _ = multiview.train_product_manifold(
        hyper, eucl, [("a", "b")], epochs=3, lr=0.1
    )
    after = np.linalg.norm(h_new["a"] - h_new["b"]) ** 2
    assert after < before


@pytest.mark.heavy
def test_cca_align_and_load(monkeypatch, tmp_path):
    monkeypatch.setattr(multiview, "CCA", DummyCCA)
    n2v = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    gw = {"a": [0.5, 0.5], "b": [-0.5, -0.5]}
    latent = multiview.cca_align(n2v, gw, n_components=1, path=tmp_path / "cca.pkl")
    assert set(latent) == {"a", "b"}
    W1, W2 = multiview.load_cca(tmp_path / "cca.pkl")
    assert W1.shape == (2, 1) and W2.shape == (2, 1)


@pytest.mark.heavy
def test_hybrid_contrastive_and_meta(monkeypatch):
    # stub sklearn dependency so meta_autoencoder runs without the real package
    sk_stub = types.SimpleNamespace(decomposition=types.SimpleNamespace(PCA=DummyPCA))
    monkeypatch.setitem(sys.modules, "sklearn", sk_stub)
    monkeypatch.setitem(sys.modules, "sklearn.decomposition", sk_stub.decomposition)
    n2v = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    gw = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    hyp = {"a": [0.1, 0.0], "b": [0.0, 0.1]}
    score = multiview.hybrid_score(
        n2v["a"], n2v["b"], gw["a"], gw["b"], hyp["a"], hyp["b"]
    )
    assert score != 0.0
    loss = multiview.multiview_contrastive_loss(n2v, gw, hyp, tau=0.5)
    assert loss >= 0.0
    latent, recon = multiview.meta_autoencoder(n2v, gw, hyp, bottleneck=1)
    assert len(latent) == 2
    assert recon["a"].shape == (6,)
