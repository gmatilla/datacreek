import numpy as np

from datacreek.analysis import betti_curve


def test_betti_curve_exposed():
    diag = np.array([[0.0, 1.0], [0.5, 1.5]])
    ts = np.linspace(0.0, 1.5, 4)
    curve = betti_curve(diag, ts)
    assert curve.shape == (4,)
    assert np.all(curve >= 0)
