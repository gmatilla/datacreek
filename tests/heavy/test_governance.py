import numpy as np
import pytest

from datacreek.analysis import governance


@pytest.mark.heavy
def test_alignment_correlation_perfect():
    emb1 = {"a": [1, 0], "b": [0, 1]}
    emb2 = {"a": [2, 0], "b": [0, 2]}
    corr = governance.alignment_correlation(emb1, emb2)
    assert pytest.approx(corr) == 1.0


@pytest.mark.heavy
def test_average_hyperbolic_radius_simple():
    emb = {"a": [0.0, 0.0], "b": [0.6, 0.0]}
    radius = governance.average_hyperbolic_radius(emb)
    assert pytest.approx(radius) == np.arctanh(0.6) / 2


@pytest.mark.heavy
def test_scale_bias_wasserstein_and_metrics():
    e1 = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    e2 = {"a": [1.0, 0.0], "b": [0.0, 0.0]}
    dist = governance.scale_bias_wasserstein(e1, e2)
    # zero is possible when norm distributions match
    assert dist >= 0
    metrics = governance.governance_metrics(e1, e1, e2)
    assert metrics["bias_wasserstein"] == pytest.approx(dist)
    assert 0 <= metrics["alignment_corr"] <= 1


@pytest.mark.heavy
def test_mitigate_bias_wasserstein_balances_groups():
    emb = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 1.0]}
    groups = {"a": "x", "b": "y", "c": "y"}
    out = governance.mitigate_bias_wasserstein(emb, groups)
    assert set(out) == set(groups)
    norms = {g: [] for g in set(groups.values())}
    for node, vec in out.items():
        norms[groups[node]].append(np.linalg.norm(vec))
    means = {g: np.mean(v) for g, v in norms.items()}
    assert pytest.approx(means["x"]) == means["y"]
