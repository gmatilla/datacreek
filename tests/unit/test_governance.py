import numpy as np
import pytest

from datacreek.analysis import governance


def test_alignment_correlation_perfect():
    emb1 = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    emb2 = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    corr = governance.alignment_correlation(emb1, emb2)
    assert pytest.approx(corr, abs=1e-6) == 1.0


def test_alignment_correlation_no_common():
    with pytest.raises(ValueError):
        governance.alignment_correlation({"a": [0]}, {"b": [0]})


def test_average_hyperbolic_radius():
    emb = {"a": [0.0, 0.0], "b": [0.5, 0.0]}
    radius = governance.average_hyperbolic_radius(emb)
    expected = 0.5 * np.arctanh(0.5)  # mean of radii 0 and arctanh(0.5)
    assert pytest.approx(radius, abs=1e-6) == expected


def test_scale_bias_wasserstein():
    e1 = {"a": [0.0, 0.0], "b": [1.0, 0.0]}
    e2 = {"a": [0.5, 0.0], "b": [1.5, 0.0]}
    e3 = {"a": [2.0, 0.0], "b": [2.0, 0.0]}
    max_w = governance.scale_bias_wasserstein(e1, e2, e3)
    # expected value depends on the wasserstein_distance implementation
    dists = [
        governance.wasserstein_distance([0, 1], [0.5, 1.5]),
        governance.wasserstein_distance([0, 1], [2, 2]),
        governance.wasserstein_distance([0.5, 1.5], [2, 2]),
    ]
    assert pytest.approx(max_w, abs=1e-6) == max(dists)


def test_governance_metrics_combines_functions(monkeypatch):
    calls = {}

    def fake_corr(x, y):
        calls["corr"] = True
        return 0.1

    def fake_rad(x):
        calls["rad"] = True
        return 0.2

    def fake_bias(*a):
        calls["bias"] = True
        return 0.3

    monkeypatch.setattr(governance, "alignment_correlation", fake_corr)
    monkeypatch.setattr(governance, "average_hyperbolic_radius", fake_rad)
    monkeypatch.setattr(governance, "scale_bias_wasserstein", fake_bias)

    res = governance.governance_metrics({}, {}, {})
    assert res == {
        "alignment_corr": 0.1,
        "hyperbolic_radius": 0.2,
        "bias_wasserstein": 0.3,
    }
    assert calls == {"corr": True, "rad": True, "bias": True}


def test_mitigate_bias_wasserstein():
    embs = {"A": np.array([1.0, 0.0]), "B": np.array([2.0, 0.0])}
    groups = {"A": "g1", "B": "g2"}
    res = governance.mitigate_bias_wasserstein(embs, groups)
    # both norms rescaled to global mean of 1.5
    assert np.allclose(res["A"], [1.5, 0.0])
    assert np.allclose(res["B"], [1.5, 0.0])
