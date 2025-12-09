import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np

from datacreek.analysis.autotune import svgp_ei_propose


def test_svgp_ei_propose_bounds():
    params = [[0.0, 0.0], [1.0, 1.0]]
    scores = [1.0, 0.8]
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    x = svgp_ei_propose(params, scores, bounds, m=2, n_samples=32)
    assert len(x) == 2
    assert 0.0 <= x[0] <= 1.0
    assert 0.0 <= x[1] <= 1.0
