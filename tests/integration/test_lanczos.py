import numpy as np

from datacreek.analysis.fractal import lanczos_lmax


def test_lanczos_lmax_approx():
    L = np.array([[2.0, -1.0], [-1.0, 2.0]])
    val = lanczos_lmax(L, iters=5)
    assert abs(val - 3.0) < 1e-2
