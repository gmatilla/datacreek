import builtins

import numpy as np

from datacreek.utils import retrieval


class FakeVec:
    """Simple count-based vectorizer used for testing."""

    def fit(self, texts):
        self.vocab = sorted({w for t in texts for w in t.split()})
        return self

    def transform(self, texts):
        rows = []
        for t in texts:
            counts = {w: t.split().count(w) for w in self.vocab}
            rows.append([counts[w] for w in self.vocab])
        arr = np.array(rows, dtype=float)

        # emulate sparse matrix API of scikit-learn
        class Mat:
            def __init__(self, array):
                self.array = array
                self.shape = array.shape

            def toarray(self):
                return self.array

        return Mat(arr)


class FakeNN:
    """NearestNeighbors replacement using simple L2 distance."""

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


def test_basic_search_and_neighbors(monkeypatch):
    monkeypatch.setattr(retrieval, "TfidfVectorizer", FakeVec)
    monkeypatch.setattr(retrieval, "NearestNeighbors", FakeNN)
    monkeypatch.setattr(retrieval, "np", np)

    idx = retrieval.EmbeddingIndex()
    idx.add("1", "hello world")
    idx.add("2", "world of python")
    idx.add("3", "greetings earth")
    idx.build()

    # search should return indices of the most relevant chunks
    res = idx.search("hello world", k=2)
    assert res[0] == 0
    nn = idx.nearest_neighbors(k=1)
    assert set(nn.keys()) == {"1", "2", "3"}
    # distance information is available when requested
    dists = idx.nearest_neighbors(k=1, return_distances=True)
    assert isinstance(dists["1"][0][1], float)
    # helper accessors
    assert idx.get_text(0) == "hello world"
    assert idx.get_id(0) == "1"
    # embedding helpers
    tmat = idx.transform(["hello world"])
    assert tmat.shape[0] == 1
    assert idx.embed("hello world").shape == (tmat.shape[1],)


def test_remove_and_rebuild(monkeypatch):
    monkeypatch.setattr(retrieval, "TfidfVectorizer", FakeVec)
    monkeypatch.setattr(retrieval, "NearestNeighbors", FakeNN)
    monkeypatch.setattr(retrieval, "np", np)

    idx = retrieval.EmbeddingIndex()
    idx.add("a", "x y z")
    idx.add("b", "x y w")
    idx.build()
    idx.remove("a")
    idx.build()
    assert idx.ids == ["b"]
    assert idx.search("x")[0] == 0


def test_fallback_count_vectors(monkeypatch):
    class FakeNN:
        def __init__(self, metric="cosine"):
            self.X = None

        def fit(self, X):
            self.X = np.array(X)

        def kneighbors(self, X, n_neighbors):
            X = np.array(X)
            d = np.linalg.norm(self.X - X[:, None], axis=2)
            idx = np.argsort(d, axis=1)[:, :n_neighbors]
            dist = np.take_along_axis(d, idx, axis=1)
            return dist, idx

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError
        return original_import(name, *args, **kwargs)

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
    # with no vectorizer, transform returns empty embedding
    assert idx.transform(["alpha"]).size == 0


def test_search_empty_and_remove_missing():
    idx = retrieval.EmbeddingIndex()
    assert idx.search("foo") == []
    assert idx.nearest_neighbors() == {}
    idx.remove("missing")  # should not raise
    assert idx.ids == []


def test_use_hnsw(monkeypatch):
    monkeypatch.setattr(retrieval, "TfidfVectorizer", FakeVec)
    monkeypatch.setattr(retrieval, "NearestNeighbors", FakeNN)
    monkeypatch.setattr(retrieval, "np", np)

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
            d = np.linalg.norm(self.data - np.array(query)[:, None], axis=2)
            idx = np.argsort(d, axis=1)[:, :k]
            dist = np.take_along_axis(d, idx, axis=1)
            return idx, dist

    monkeypatch.setattr(retrieval, "hnswlib", type("H", (), {"Index": FakeHnsw}))

    idx = retrieval.EmbeddingIndex(use_hnsw=True)
    idx.add("x", "zero one")
    idx.add("y", "zero two")
    idx.build()
    res = idx.nearest_neighbors(k=1)
    assert res["x"][0] == "y"
    assert idx._hnsw is not None


def test_search_hnsw(monkeypatch):
    """Ensure ``search`` uses the HNSW index when available."""
    monkeypatch.setattr(retrieval, "TfidfVectorizer", FakeVec)
    monkeypatch.setattr(retrieval, "NearestNeighbors", FakeNN)
    monkeypatch.setattr(retrieval, "np", np)

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
            d = np.linalg.norm(self.data - np.array(query)[:, None], axis=2)
            idx = np.argsort(d, axis=1)[:, :k]
            return idx, d

    monkeypatch.setattr(retrieval, "hnswlib", type("H", (), {"Index": FakeHnsw}))

    idx = retrieval.EmbeddingIndex(use_hnsw=True)
    idx.add("x", "zero one")
    idx.add("y", "zero two")
    idx.build()
    assert idx.search("zero", k=1)[0] in {0, 1}


def test_import_success(monkeypatch):
    """Reload module when optional imports succeed to cover import branch."""
    import importlib
    import sys

    fake_hnsw = type("F", (), {})
    monkeypatch.setitem(sys.modules, "hnswlib", fake_hnsw)
    monkeypatch.setitem(
        sys.modules,
        "sklearn.feature_extraction.text",
        type("M", (), {"TfidfVectorizer": FakeVec}),
    )
    monkeypatch.setitem(
        sys.modules,
        "sklearn.neighbors",
        type("N", (), {"NearestNeighbors": FakeNN}),
    )

    mod = importlib.reload(retrieval)
    # force dynamic import branch inside ``build``
    mod.TfidfVectorizer = None
    mod.NearestNeighbors = None
    mod.np = None
    idx = mod.EmbeddingIndex()
    idx.add("i", "dummy text")
    idx.build()
    assert isinstance(mod.TfidfVectorizer(), FakeVec)

    # reload again to restore original module state
    importlib.reload(retrieval)
