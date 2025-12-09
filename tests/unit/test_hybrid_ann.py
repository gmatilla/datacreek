import importlib
import types

import numpy as np
import pytest

import datacreek.analysis.hybrid_ann as ha


def reload_module(monkeypatch):
    """Reload module to apply monkeypatches."""
    importlib.reload(ha)


def test_expected_recall():
    assert ha.expected_recall(2, 8, 4) == pytest.approx(1 - (1 - 2 / 8) ** 4)
    assert ha.expected_recall(0, 10, 2) == 0.0


def test_choose_nprobe_multi():
    assert ha.choose_nprobe_multi(0, 0) == 1
    assert ha.choose_nprobe_multi(256, 16) == 4


def test_functions_require_faiss(monkeypatch):
    monkeypatch.setattr(ha, "faiss", None)
    with pytest.raises(RuntimeError):
        ha.load_ivfpq_cpu("index.bin", 10)
    xb = np.random.rand(10, 5).astype("float32")
    xq = np.random.rand(1, 5).astype("float32")
    with pytest.raises(RuntimeError):
        ha.rerank_pq(xb, xq)
    with pytest.raises(RuntimeError):
        ha.search_hnsw_pq(xb, xq)
