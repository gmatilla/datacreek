import numpy as np

from datacreek.analysis.autotune import svgp_ei_propose


def test_svgp_switches_kernel(monkeypatch):
    called = {}

    class DummyGPR:
        def __init__(self, *, kernel=None, **_):
            called["kernel"] = kernel

        def fit(self, X, y):
            pass

        def predict(self, X, return_std=True):
            return np.array([0.0]), np.array([1.0])

    monkeypatch.setattr("sklearn.gaussian_process.GaussianProcessRegressor", DummyGPR)
    params = [[i] for i in range(12)]
    scores = [float(i) for i in range(12)]
    bounds = [(0.0, 1.0)]
    monkeypatch.setattr(np, "var", lambda arr: 1.0)
    svgp_ei_propose(params, scores, bounds, m=5, n_samples=4)
    from sklearn.gaussian_process.kernels import Matern

    assert isinstance(called["kernel"].k2, Matern)
