import numpy as np

from scripts.train_alpha import train_alpha


def test_train_alpha_projected_simplex():
    def grad_fn(alpha):
        return np.array([1.0, 0.0, -1.0])

    alpha = train_alpha(np.array([0.3, 0.3, 0.4]), grad_fn, lr=0.5, steps=5)
    assert np.isclose(alpha.sum(), 1.0)
    assert np.all(alpha >= 0)
    assert alpha[0] > alpha[2]
