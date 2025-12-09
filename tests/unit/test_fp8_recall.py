import numpy as np

from datacreek.analysis.compression import fp8_dequantize, fp8_quantize


def _topk_neighbors(x: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the ``k`` nearest neighbors for each vector in ``x``.

    Distances are computed using the Euclidean norm. The diagonal is ignored to
    avoid self-matches.
    """
    dists = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.argsort(dists, axis=1)[:, :k]


def test_fp8_recall_at_k() -> None:
    """Quantization should preserve nearest neighbors up to 0.5% recall loss.

    The FP8 scheme stores each embedding ``e`` as ``q`` and a single scale ``s``
    such that ``e ≈ q * s / 127``. We check that the average recall@5 between the
    original vectors and their quantized reconstruction differs by at most 0.005.
    """
    dim = 16
    data = np.vstack([np.eye(dim), -np.eye(dim)]).astype(np.float32)
    q, scale = fp8_quantize(data)
    restored = fp8_dequantize(q, scale)

    k = 5
    top_exact = _topk_neighbors(data, k)
    top_restored = _topk_neighbors(restored, k)

    recall = [
        len(set(top_exact[i]).intersection(top_restored[i])) / k
        for i in range(len(data))
    ]
    mean_recall = float(np.mean(recall))
    assert 1.0 - mean_recall <= 0.005
