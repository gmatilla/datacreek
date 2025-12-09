import numpy as np

from datacreek.analysis.hypergraph import hyperedge_attention_scores


def test_hyperedge_attention_scores():
    edges = [[0, 1], [1, 2, 3]]
    feats = np.eye(4)
    scores = hyperedge_attention_scores(edges, feats, seed=0)
    assert scores.shape == (2,)
    assert scores.dtype == float
