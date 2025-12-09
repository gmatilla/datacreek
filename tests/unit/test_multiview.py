import os
import pickle
import sys
import types

import networkx as nx
import numpy as np
import pytest

from datacreek.analysis import multiview


def test_product_embedding_concatenation():
    hyp = {"a": [1.0, 2.0], "b": [3.0, 4.0]}
    eu = {"a": [5.0, 6.0], "b": [7.0, 8.0]}
    result = multiview.product_embedding(hyp, eu)
    assert set(result) == {"a", "b"}
    assert np.allclose(result["a"], [1.0, 2.0, 5.0, 6.0])
    assert np.allclose(result["b"], [3.0, 4.0, 7.0, 8.0])


def test_train_product_manifold_updates():
    h = {"a": [0.1], "b": [0.2]}
    e = {"a": [0.0], "b": [0.1]}
    ctx = [("a", "b")]
    new_h, new_e = multiview.train_product_manifold(h, e, ctx, epochs=1, lr=0.5)
    assert not np.allclose(h["a"], new_h["a"])
    assert np.linalg.norm(new_h["a"]) < 1.0
    assert not np.allclose(e["a"], new_e["a"])
    assert np.linalg.norm(new_e["a"]) < 1.0


def test_aligned_cca_and_persistence(monkeypatch, tmp_path):
    class DummyCCA:
        def __init__(self, n_components=1):
            self.n_components = n_components

        def fit_transform(self, X, Y):
            self.x_weights_ = np.ones((X.shape[1], self.n_components))
            self.y_weights_ = np.ones((Y.shape[1], self.n_components))
            return (X + Y) / 2, (X + Y) / 2

    monkeypatch.setattr(multiview, "CCA", DummyCCA)
    n2v = {"n": [1.0, 0.0]}
    gw = {"n": [0.0, 1.0]}
    latent, cca = multiview.aligned_cca(n2v, gw, n_components=1)
    assert isinstance(cca, DummyCCA)
    assert np.allclose(latent["n"], [0.5, 0.5])
    path = tmp_path / "cca.pkl"
    out = multiview.cca_align(n2v, gw, n_components=1, path=str(path))
    assert path.exists() and "n" in out
    W1, W2 = multiview.load_cca(str(path))
    assert W1.shape[1] == W2.shape[1] == 1


def _manual_hybrid(n2v_u, n2v_q, gw_u, gw_q, hyp_u, hyp_q, gamma=0.5, eta=0.25):
    def _cos(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
        return float(np.dot(a, b) / denom)

    def _poincare(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        nx_ = np.linalg.norm(x)
        ny_ = np.linalg.norm(y)
        diff = np.linalg.norm(x - y)
        arg = 1.0 + 2 * diff * diff / ((1 - nx_ * nx_) * (1 - ny_ * ny_) + 1e-9)
        return float(np.arccosh(max(1.0, arg)))

    cos_n2v = _cos(n2v_u, n2v_q)
    cos_gw = _cos(gw_u, gw_q)
    dist_h = _poincare(hyp_u, hyp_q)
    return gamma * cos_n2v + eta * (1.0 - dist_h) + (1 - gamma - eta) * (1 - cos_gw)


def test_hybrid_score_matches_formula():
    n2v_u = [1.0, 0.0]
    n2v_q = [0.5, 0.5]
    gw_u = [0.0, 1.0]
    gw_q = [0.0, 1.0]
    hyp_u = [0.2, 0.1]
    hyp_q = [0.3, 0.0]
    expected = _manual_hybrid(n2v_u, n2v_q, gw_u, gw_q, hyp_u, hyp_q)
    result = multiview.hybrid_score(n2v_u, n2v_q, gw_u, gw_q, hyp_u, hyp_q)
    assert pytest.approx(result, rel=1e-6) == expected


def test_multiview_contrastive_loss():
    n2v = {0: [1.0, 0.0], 1: [0.0, 1.0]}
    gw = {0: [0.0, 1.0], 1: [1.0, 0.0]}
    hyp = {0: [0.1, 0.1], 1: [0.2, 0.0]}
    loss = multiview.multiview_contrastive_loss(n2v, gw, hyp, tau=1.0)
    assert loss > 0


def test_meta_autoencoder(monkeypatch):
    class DummyPCA:
        def __init__(self, n_components):
            self.n_components = n_components

        def fit_transform(self, X):
            self._dim = X.shape[1]
            return X[:, : self.n_components]

        def inverse_transform(self, Z):
            pad = self._dim - Z.shape[1]
            return np.hstack([Z, np.zeros((Z.shape[0], pad))])

    monkeypatch.setitem(
        sys.modules, "sklearn.decomposition", types.SimpleNamespace(PCA=DummyPCA)
    )
    n2v = {0: [1.0, 0.0], 1: [0.0, 1.0]}
    gw = {0: [0.0, 1.0], 1: [1.0, 0.0]}
    hyp = {0: [0.5, 0.0], 1: [0.0, 0.5]}
    latent, recon = multiview.meta_autoencoder(n2v, gw, hyp, bottleneck=2)
    assert set(latent) == set(recon) == {0, 1}
    assert latent[0].shape[0] == 2
