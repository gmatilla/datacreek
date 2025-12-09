import math

import numpy as np
import pytest

from datacreek.analysis.hypergraph import (
    hyper_adamic_adar_scores,
    hyper_sagnn_embeddings,
    hyper_sagnn_head_drop_embeddings,
)
from datacreek.core.knowledge_graph import KnowledgeGraph


def test_hyper_sagnn_embeddings_shape():
    edges = [[0, 1], [1, 2, 3]]
    feats = np.arange(12).reshape(4, 3).astype(float)
    emb = hyper_sagnn_embeddings(edges, feats, embed_dim=4, seed=0)
    assert emb.shape == (2, 4)


def test_dataset_hyper_sagnn_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_document("d", source="s")
    ds.add_chunk("d", "a", "x")
    ds.add_chunk("d", "b", "y")
    ds.add_chunk("d", "c", "z")
    ds.add_hyperedge("h1", ["a", "b", "c"])

    for idx, n in enumerate(["a", "b", "c"]):
        ds.graph.graph.nodes[n]["embedding"] = [float(idx), 0.0]

    result = ds.compute_hyper_sagnn_embeddings(embed_dim=2, seed=0)
    assert "h1" in result
    assert len(result["h1"]) == 2
    assert any(e.operation == "compute_hyper_sagnn_embeddings" for e in ds.events)

    result_hd = ds.compute_hyper_sagnn_head_drop_embeddings(
        num_heads=2, threshold=0.0, seed=0
    )
    assert "h1" in result_hd
    assert len(result_hd["h1"]) == 1
    assert any(
        e.operation == "compute_hyper_sagnn_head_drop_embeddings" for e in ds.events
    )


def test_hyper_sagnn_head_drop():
    edges = [[0, 1, 2], [2, 3]]
    feats = np.random.RandomState(0).randn(4, 4)
    emb = hyper_sagnn_head_drop_embeddings(
        edges, feats, num_heads=2, threshold=0.0, seed=0
    )
    assert emb.shape[0] == 2
    assert emb.shape[1] == 2


def test_hyper_adamic_adar_simple():
    edges = [["a", "b"], ["a", "c"]]
    scores = hyper_adamic_adar_scores(edges)
    w = 0.0  # log(1) -> 0, score defaults to 0
    assert scores[("a", "b")] == pytest.approx(w)
    assert scores[("a", "c")] == pytest.approx(w)
    assert ("b", "c") not in scores


def test_hyper_adamic_adar_on_graph():
    kg = KnowledgeGraph()
    kg.add_document("d", source="s")
    for cid in ("c1", "c2", "c3"):
        kg.add_chunk("d", cid, "t")
    kg.add_hyperedge("h1", ["c1", "c2"])
    kg.add_hyperedge("h2", ["c1", "c3"])
    scores = kg.hyper_adamic_adar_scores()
    assert ("c1", "c2") in scores


def test_hyper_adamic_adar_triangle():
    edges = [["x", "y", "z"]]
    scores = hyper_adamic_adar_scores(edges)
    expected = 1.0 / math.log(2)
    assert scores[("x", "y")] == pytest.approx(expected)
    assert scores[("x", "z")] == pytest.approx(expected)
    assert scores[("y", "z")] == pytest.approx(expected)
    for pair in [("x", "y"), ("x", "z"), ("y", "z")]:
        assert abs(scores[pair] - 1 / np.log(2)) < 1e-6
