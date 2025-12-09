"""Tests for topological data analysis helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from datacreek import tda


def test_persistence_minhash_uses_gudhi(monkeypatch):
    """The persistence sketch feeds pairs through the MinHash pipeline."""

    # Stub GUDHI components to avoid heavy computation
    class DummyST:
        def persistence(self, p: int = 2):  # pragma: no cover - simple stub
            return [(0.0, 1.0), (0.0, 2.0)]

    class DummyRips:
        def __init__(self, points):
            pass

        def create_simplex_tree(self, max_dimension: int = 2):  # pragma: no cover
            return DummyST()

    monkeypatch.setattr(tda, "gudhi", SimpleNamespace(RipsComplex=DummyRips))
    sig = tda.persistence_minhash(np.array([[0.0, 0.0]]))
    assert isinstance(sig, bytes) and len(sig) == 64


def test_select_best_lens():
    """The lens with trustworthiness above the threshold is chosen."""

    emb = np.array([[0, 0], [1, 1], [2, 2]])
    good = emb.copy()
    bad = emb[::-1]
    choice = tda.select_best_lens(
        emb, {"good": good, "bad": bad}, threshold=0.95, n_neighbors=2
    )
    assert choice == "good"
