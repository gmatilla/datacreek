import numpy as np

from datacreek.analysis.governance import (
    alignment_correlation,
    average_hyperbolic_radius,
    governance_metrics,
    mitigate_bias_wasserstein,
    scale_bias_wasserstein,
)
from datacreek.core.dataset import DatasetBuilder, DatasetType


def test_governance_metrics_functions():
    n2v = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    gw = {"a": [1.0, 1.0], "b": [-1.0, -1.0]}
    hyp = {"a": [0.2, 0.0], "b": [0.0, 0.2]}
    corr = alignment_correlation(n2v, gw)
    assert -1.0 <= corr <= 1.0
    radius = average_hyperbolic_radius(hyp)
    assert radius > 0.0
    bias = scale_bias_wasserstein(n2v, gw, hyp)
    assert bias >= 0.0
    metrics = governance_metrics(n2v, gw, hyp)
    assert set(metrics) == {"alignment_corr", "hyperbolic_radius", "bias_wasserstein"}


def test_mitigate_bias_wasserstein_function():
    emb = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 1.0]}
    groups = {"a": "g1", "b": "g2", "c": "g2"}
    res = mitigate_bias_wasserstein(emb, groups)
    g1_mean = np.linalg.norm(res["a"])
    g2_mean = np.mean([np.linalg.norm(res[n]) for n in ["b", "c"]])
    assert abs(g1_mean - g2_mean) < 1e-6


def test_dataset_governance_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_entity("e1", "A")
    ds.add_entity("e2", "B")
    ds.graph.graph.nodes["e1"].update(
        {
            "embedding": [1.0, 0.0],
            "graphwave_embedding": [0.5, 0.5],
            "poincare_embedding": [0.1, 0.0],
        }
    )
    ds.graph.graph.nodes["e2"].update(
        {
            "embedding": [0.0, 1.0],
            "graphwave_embedding": [-0.5, -0.5],
            "poincare_embedding": [0.0, 0.1],
        }
    )
    metrics = ds.governance_metrics()
    assert "alignment_corr" in metrics and "hyperbolic_radius" in metrics


def test_dataset_mitigate_bias_wrapper():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.add_entity("u1", "A")
    ds.graph.graph.nodes["u1"].update({"embedding": [1.0, 0.0]})
    ds.add_entity("u2", "B")
    ds.graph.graph.nodes["u2"].update({"embedding": [0.0, 1.0]})
    res = ds.mitigate_bias_wasserstein({"u1": "g1", "u2": "g2"})
    assert set(res) == {"u1", "u2"}


def test_sheaf_consistency_score_function():
    g = DatasetBuilder(DatasetType.TEXT).graph.graph
    g.add_edge("a", "b", sheaf_sign=1)
    from datacreek.analysis.sheaf import sheaf_consistency_score

    score = sheaf_consistency_score(g)
    assert 0.0 <= score <= 1.0
