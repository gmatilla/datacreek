import types

import networkx as nx
import numpy as np
import pytest

import datacreek.analysis.index as idx


class DummyTimer:
    def __init__(self, times):
        self._it = iter(times)

    def monotonic(self):
        return next(self._it)


class Flat:
    def __init__(self, d):
        self.d = d

    def add(self, xb):
        self.xb = xb

    def search(self, xq, k):
        return None, np.tile(np.arange(k), (xq.shape[0], 1))


class HNSWFlat(Flat):
    def __init__(self, d, m):
        super().__init__(d)
        self.hnsw = types.SimpleNamespace(efSearch=0)


@pytest.mark.heavy
def test_search_with_fallback(monkeypatch):
    xb = np.eye(2, dtype=np.float32)
    xq = xb[:1]
    timer = DummyTimer([0.0, 0.2, 0.3, 0.4])
    monkeypatch.setattr(idx, "time", timer)
    monkeypatch.setattr(
        idx, "faiss", types.SimpleNamespace(IndexFlatIP=Flat, IndexHNSWFlat=HNSWFlat)
    )
    monkeypatch.setattr(idx, "np", np)
    out_idx, lat, index = idx.search_with_fallback(xb, xq, k=1, latency_threshold=0.1)
    assert lat == pytest.approx(0.1)
    assert isinstance(index, HNSWFlat)
    assert out_idx.tolist() == [0]


@pytest.mark.heavy
def test_recall10_basic(monkeypatch):
    g = nx.path_graph(3)
    for n in g:
        g.nodes[n]["embedding"] = [1.0, 0.0] if n != 2 else [-1.0, 0.0]
        g.nodes[n]["graphwave_embedding"] = g.nodes[n]["embedding"]
        g.nodes[n]["poincare_embedding"] = [0.0, 0.0]
    monkeypatch.setattr(idx, "update_metric", lambda *a, **k: None)
    rec = idx.recall10(g, [0], {0: [1]}, gamma=1.0, eta=0.0)
    assert rec == pytest.approx(0.1)
    assert g.graph["recall10"] == rec
