try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency missing
    np = None
import pytest

from datacreek.analysis.compression import prune_fractalnet


def test_prune_fractalnet_basic():
    if np is None:
        pytest.skip("numpy not installed")
    w = np.array([1, -2, 3, -4, 5, -6, 7, -8, 9, -10], dtype=float)
    pruned = prune_fractalnet(w, ratio=0.5)
    assert np.count_nonzero(pruned) == 5
    assert pruned.shape == w.shape


def test_dataset_prune_wrapper():
    if np is None:
        pytest.skip("numpy not installed")
    try:
        from datacreek.core.dataset import DatasetBuilder
    except Exception:
        pytest.skip("DatasetBuilder unavailable")
    db = DatasetBuilder()
    w = np.linspace(-1, 1, 20)
    out = db.prune_fractalnet_weights(w, ratio=0.25)
    assert np.count_nonzero(out) == 5
