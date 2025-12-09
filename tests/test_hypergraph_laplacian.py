import numpy as np

from datacreek.hypergraph import laplacian, multiplex_laplacian


def test_laplacian_row_sums_zero():
    B = np.array([[1, 0], [0, 1]])
    L = laplacian(B)
    assert np.allclose(L.sum(axis=1), 0)


def test_multiplex_laplacian_composition():
    B1 = np.array([[1, 0], [1, 1]])
    B2 = np.array([[0, 1], [1, 0]])
    alpha = np.array([0.5, 0.5])
    L_multi = multiplex_laplacian([B1, B2], alpha)
    expected = 0.5 * laplacian(B1) + 0.5 * laplacian(B2)
    assert np.allclose(L_multi, expected)
