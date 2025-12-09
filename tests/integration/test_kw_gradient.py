import math

from datacreek.analysis.autotune import kw_gradient


def test_kw_gradient_quadratic():
    f = lambda x: (x - 1.0) ** 2
    g = kw_gradient(f, 1.0, h=0.5, n=20)
    assert abs(g) < 0.2
