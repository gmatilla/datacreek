import builtins
import types

import numpy as np
import pytest

from datacreek.utils import retrieval


class FakeVec:
    def fit(self, texts):
        self.vocab = sorted({w for t in texts for w in t.split()})
        return self

    def transform(self, texts):
        rows = []
        for t in texts:
            counts = {w: t.split().count(w) for w in self.vocab}
            rows.append([counts[w] for w in self.vocab])
        arr = np.array(rows, dtype=float)

        class Mat:
            def __init__(self, array):
                self.array = array
                self.shape = array.shape

            def toarray(self):
                return self.array

        return Mat(arr)


class FakeNN:
    def __init__(self, *a, **kw):
        pass

    def fit(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        self.X = np.array(X)

    def kneighbors(self, X, n_neighbors):
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.array(X)
        d = np.linalg.norm(self.X - X[:, None], axis=2)
        idx = np.argsort(d, axis=1)[:, :n_neighbors]
        dist = np.take_along_axis(d, idx, axis=1)
        return dist, idx


class FakeHnsw:
    def __init__(self, space="cosine", dim=0):
        self.data = None

    def init_index(self, max_elements, ef_construction=100, M=16):
        pass

    def add_items(self, data, ids):
        self.data = np.array(data)

    def set_ef(self, ef):
        pass

    def knn_query(self, query, k):
        q = np.array(query)
        d = np.linalg.norm(self.data - q[:, None], axis=2)
        idx = np.argsort(d, axis=1)[:, :k]
        dist = np.take_along_axis(d, idx, axis=1)
        return idx, dist


@pytest.mark.heavy
def test_hnsw_and_helpers(monkeypatch):
    monkeypatch.setattr(retrieval, "TfidfVectorizer", FakeVec)
    monkeypatch.setattr(retrieval, "NearestNeighbors", FakeNN)
    monkeypatch.setattr(retrieval, "np", np)
    monkeypatch.setattr(retrieval, "hnswlib", types.SimpleNamespace(Index=FakeHnsw))

    idx = retrieval.EmbeddingIndex(use_hnsw=True)
    for i, t in enumerate(["hello world", "world of python", "greetings earth"]):
        idx.add(str(i), t)
    idx.build()

    # hnsw path
    res = idx.search("hello world", k=2)
    assert res[0] == 0
    nn = idx.nearest_neighbors(k=1)
    assert set(nn.keys()) == {"0", "1", "2"}
    assert idx.get_text(0) == "hello world"
    assert idx.get_id(0) == "0"
    tmat = idx.transform(["hello world"])
    assert tmat.shape[0] == 1
    assert idx.embed("hello world").shape == (tmat.shape[1],)


@pytest.mark.heavy
def test_fallback_without_sklearn(monkeypatch):
    # simulate missing sklearn
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(retrieval, "TfidfVectorizer", None)
    monkeypatch.setattr(retrieval, "NearestNeighbors", FakeNN)
    monkeypatch.setattr(retrieval, "np", None)

    idx = retrieval.EmbeddingIndex()
    idx.add("a", "alpha beta")
    idx.add("b", "alpha gamma")
    idx.build()
    res = idx.nearest_neighbors(k=1)
    assert set(res.keys()) == {"a", "b"}
    assert idx.transform(["alpha"]).size == 0


@pytest.mark.heavy
def test_non_hnsw_flow(monkeypatch):
    monkeypatch.setattr(retrieval, "TfidfVectorizer", FakeVec)
    monkeypatch.setattr(retrieval, "NearestNeighbors", FakeNN)
    monkeypatch.setattr(retrieval, "np", np)
    idx = retrieval.EmbeddingIndex()
    for i, t in enumerate(["a b", "b c", "c d"]):
        idx.add(str(i), t)
    idx.build()
    nn = idx.nearest_neighbors(k=1)
    assert set(nn.keys()) == {"0", "1", "2"}
    search_idx = idx.search("a b", k=1)
    assert search_idx[0] == 0
    idx.remove("1")
    idx.build()
    assert len(idx.ids) == 2
