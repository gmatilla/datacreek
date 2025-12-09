import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np

from datacreek.core.dataset import DatasetBuilder, DatasetType
from datacreek.core.knowledge_graph import KnowledgeGraph


def test_compute_graphwave_embeddings_chebyshev_wrapper(monkeypatch):
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "x")
    ds.add_entity("e1", "E")
    ds.link_entity("c1", "e1")
    called = {}

    def fake(self, scales, num_points=10, *, chebyshev_order=None):
        called["order"] = chebyshev_order

    monkeypatch.setattr(ds.graph.__class__, "compute_graphwave_embeddings", fake)
    ds.compute_graphwave_embeddings(scales=[0.5], num_points=4, chebyshev_order=3)
    assert called["order"] == 3


def test_svgp_ei_propose_wrappers(monkeypatch):
    kg = KnowledgeGraph()
    called = {}

    def fake(params, scores, bounds, *, m=100, n_samples=256):
        called["args"] = (params, scores, bounds, m, n_samples)
        return np.array([0.1, 0.2])

    monkeypatch.setattr("datacreek.analysis.autotune.svgp_ei_propose", fake)
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.graph = kg
    vec = ds.svgp_ei_propose(
        [([0.0, 0.0], 1.0)], [(0.0, 1.0), (0.0, 1.0)], m=10, n_samples=20
    )
    assert called["args"][3] == 10
    assert isinstance(vec, list) and len(vec) == 2
