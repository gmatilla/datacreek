import sys
import types

import networkx as nx
import numpy as np
import pytest

from datacreek.analysis import information


class DummyLR:
    """Minimal deterministic stand-in for LogisticRegression."""

    def __init__(self, max_iter=1000, n_jobs=1):
        pass

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.means = {c: X[y == c].mean(axis=0) for c in self.classes_}

    def predict_proba(self, X):
        dists = np.stack(
            [np.linalg.norm(X - self.means[c], axis=1) for c in self.classes_],
            axis=1,
        )
        probs = np.exp(-dists)
        return probs / probs.sum(axis=1, keepdims=True)


@pytest.fixture(autouse=True)
def fake_logreg(monkeypatch):
    mod = types.SimpleNamespace(LogisticRegression=DummyLR)
    monkeypatch.setitem(sys.modules, "sklearn.linear_model", mod)
    yield


def test_graph_information_bottleneck_value():
    features = {
        0: np.array([0.0, 0.0]),
        1: np.array([1.0, 1.0]),
        2: np.array([2.0, 2.0]),
    }
    labels = {0: 0, 1: 1, 2: 1}
    val = information.graph_information_bottleneck(features, labels)

    X = np.stack([features[n] for n in labels])
    y = np.array([labels[n] for n in labels])
    model = DummyLR()
    model.fit(X, y)
    probs = model.predict_proba(X)
    ce = -np.mean(np.log(probs[np.arange(len(y)), y]))
    cov = np.cov(X, rowvar=False)
    reg = 0.5 * np.log(np.linalg.det(np.eye(cov.shape[0]) + cov))
    expected = float(ce + reg)
    assert pytest.approx(val, rel=1e-6) == expected


def test_graph_information_bottleneck_missing():
    with pytest.raises(ValueError):
        information.graph_information_bottleneck({}, {0: 1})


def test_prototype_subgraph():
    g = nx.Graph([(0, 1), (1, 2), (2, 3)])
    features = {
        0: np.array([0.0, 0.0]),
        1: np.array([1.0, 1.0]),
        2: np.array([2.0, 2.0]),
        3: np.array([1.5, 1.0]),
    }
    labels = {0: 0, 1: 1, 2: 1, 3: 1}
    sub = information.prototype_subgraph(g, features, labels, 1, radius=1)
    assert sorted(sub.nodes()) == [1, 2, 3]
    assert sorted(map(tuple, sub.edges())) == [(1, 2), (2, 3)]


def test_prototype_subgraph_bad_class():
    g = nx.path_graph(3)
    features = {0: np.array([0, 0])}
    labels = {0: 0}
    with pytest.raises(ValueError):
        information.prototype_subgraph(g, features, labels, 1)


def test_mdl_and_select():
    g = nx.Graph([(0, 1), (1, 2), (2, 0), (0, 3)])
    tri = nx.Graph([(0, 1), (1, 2), (2, 0)])
    path = nx.Graph([(0, 3)])
    mdl_all = information.mdl_description_length(g, [tri, path])
    mdl_tri = information.mdl_description_length(g, [tri])
    mdl_path = information.mdl_description_length(g, [path])
    assert mdl_all == mdl_tri < mdl_path

    selected = information.select_mdl_motifs(g, [tri, path])
    assert len(selected) == 1
    assert set(selected[0].edges()) == set(tri.edges())


def test_entropy_variants():
    g = nx.Graph([(0, 1), (1, 2), (2, 0), (0, 3)])
    entropy_full = information.graph_entropy(g)
    entropy_sub = information.subgraph_entropy(g, [0, 1, 2])
    filtered = information.structural_entropy(g, tau=1)
    assert entropy_full > entropy_sub
    assert filtered > 0.0
