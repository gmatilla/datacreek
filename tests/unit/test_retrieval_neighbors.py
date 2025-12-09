import builtins
import importlib
import sys
import types

import numpy as np

import datacreek.utils.retrieval as retrieval


class DummyNN:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, mat):
        self.mat = mat

    def kneighbors(self, X, n_neighbors=1):
        dists = np.zeros((len(self.mat),))
        idxs = np.arange(len(self.mat))
        return np.tile(dists, (len(self.mat), 1)), np.tile(idxs, (len(self.mat), 1))


def setup_module(module):
    sys.modules["hnswlib"] = types.SimpleNamespace(Index=object)
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError
        return orig_import(name, *args, **kwargs)

    builtins.__import__ = fake_import
    importlib.reload(retrieval)
    builtins.__import__ = orig_import
    retrieval.NearestNeighbors = DummyNN


def teardown_module(module):
    sys.modules.pop("hnswlib", None)
    importlib.reload(retrieval)


def test_nearest_neighbors_basic():
    idx = retrieval.EmbeddingIndex()
    idx.add("1", "foo bar")
    idx.add("2", "foo baz")
    idx.add("3", "spam eggs")
    idx.build()
    nn = idx.nearest_neighbors(k=1)
    assert set(nn.keys()) == {"1", "2", "3"}
    assert all(len(v) <= 1 for v in nn.values())
