import sys
import types

import numpy as np

import datacreek.analysis.fractal_encoder as fe
from datacreek.analysis.fractal_encoder import online_pca_reduce


def test_fractal_encoder(monkeypatch):
    feats = {"a": [1.0, 2.0], "b": [0.0, 1.0]}

    class DummyRNG:
        def normal(self, scale, size):
            return np.full(size, scale)

    monkeypatch.setattr(fe.np.random, "default_rng", lambda seed=0: DummyRNG())
    out = fe.fractal_encoder(feats, embed_dim=2)
    scale = 1 / np.sqrt(2)
    expected_a = np.tanh(np.array(feats["a"]) @ np.full((2, 2), scale))
    expected_b = np.tanh(np.array(feats["b"]) @ np.full((2, 2), scale))
    assert np.allclose(out["a"], expected_a)
    assert np.allclose(out["b"], expected_b)


def test_online_pca_reduce(monkeypatch):
    class DummyIPCA:
        def __init__(self, n_components, batch_size=128):
            self.n_components = n_components

        def fit(self, X):
            self._dim = X.shape[1]

        def transform(self, X):
            return X[:, : self.n_components]

    sk_stub = types.SimpleNamespace(IncrementalPCA=DummyIPCA)
    monkeypatch.setitem(sys.modules, "sklearn.decomposition", sk_stub)
    X = np.eye(4)
    out = online_pca_reduce(X, n_components=2)
    assert out.shape == (4, 2)
