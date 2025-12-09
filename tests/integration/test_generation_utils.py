import networkx as nx
import numpy as np

from datacreek.analysis.generation import (
    apply_logit_bias,
    bias_reweighting,
    bias_wasserstein,
    sheaf_consistency_real,
    sheaf_score,
)


def test_sheaf_consistency_real_simple():
    g = nx.path_graph(3)
    for u, v in g.edges():
        g.edges[u, v]["sheaf_sign"] = 1
    score = sheaf_consistency_real(g, [0.0, 0.0, 0.0])
    assert score == 1.0


def test_bias_reweighting_adjusts(tmp_path):
    neigh = {"A": 1, "B": 0}
    global_d = {"A": 5, "B": 5}
    w = {"A": 1.0, "B": 1.0}
    out = bias_reweighting(neigh, global_d, w, threshold=0.0)
    assert out["B"] > w["B"]


def test_sheaf_score_identity():
    import numpy as np

    Delta = np.eye(2)
    score = sheaf_score([0.0, 0.0], Delta)
    assert score == 1.0


def test_bias_wasserstein_rescales():
    import numpy as np

    loc = np.array([[0.0], [1.0]], dtype=float)
    glob = np.array([[0.0], [2.0]], dtype=float)
    logits = np.array([1.0, 1.0], dtype=float)
    scaled, W = bias_wasserstein(loc, glob, logits)
    assert W >= 0.0
    assert np.allclose(scaled, logits * np.exp(-W))


def test_bias_wasserstein_majority_drop():
    """Majority group logits should decrease when local histogram is skewed."""

    import numpy as np

    loc = np.array([[9.0], [1.0]], dtype=float)
    glob = np.array([[5.0], [5.0]], dtype=float)
    logits = np.array([0.8, 0.2], dtype=float)
    scaled, W = bias_wasserstein(loc, glob, logits)
    assert W > 0.1
    assert scaled[0] < logits[0]


def test_apply_logit_bias():
    loc = np.array([[9.0], [1.0]], dtype=float)
    glob = np.array([[5.0], [5.0]], dtype=float)
    payload = {"logits": [0.9, 0.1]}

    W = apply_logit_bias(payload, loc, glob)
    assert W > 0
    assert payload["logits"][0] < 0.9
