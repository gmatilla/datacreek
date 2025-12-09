import networkx as nx
import numpy as np
import pytest

from datacreek.analysis import hypergraph


@pytest.mark.heavy
def test_hyper_sagnn_embeddings_shape():
    edges = [[0, 1, 2], [2, 3]]
    feats = np.eye(4)
    emb = hypergraph.hyper_sagnn_embeddings(edges, feats, embed_dim=2, seed=0)
    assert emb.shape == (2, 2)


@pytest.mark.heavy
def test_hyper_sagnn_head_drop_embeddings_pruning():
    edges = [[0, 1], [1, 2, 3]]
    feats = np.eye(4)
    emb = hypergraph.hyper_sagnn_head_drop_embeddings(
        edges, feats, num_heads=2, threshold=0.5, seed=1
    )
    # with high threshold some heads may drop leading to zero vectors
    assert emb.shape[1] <= feats.shape[1] // 2


@pytest.mark.heavy
def test_hyper_adamic_adar_scores():
    edges = [["a", "b", "c"], ["b", "c"], ["c"]]
    scores = hypergraph.hyper_adamic_adar_scores(edges)
    # compute expected weights manually
    weight = 1 / np.log(3 - 1)
    assert pytest.approx(scores[("a", "b")]) == weight
    assert pytest.approx(scores[("a", "c")]) == weight
    # second edge contributes zero weight due to denom <= 1
    assert pytest.approx(scores[("b", "c")]) == weight


@pytest.mark.heavy
def test_hyperedge_attention_scores_range():
    edges = [[0, 1, 2], [2, 3]]
    feats = np.eye(4)
    scores = hypergraph.hyperedge_attention_scores(edges, feats, seed=0)
    assert scores.shape == (2,)
    assert np.all(scores >= 0)
