import numpy as np
import pytest

import datacreek.analysis.drift as drift
from datacreek.analysis import baseline_mean, embedding_mmd


def test_baseline_mean():
    emb = np.array([[0.0, 1.0], [2.0, 3.0]])
    mu0 = baseline_mean(emb)
    np.testing.assert_allclose(mu0, np.array([1.0, 2.0]))


def test_embedding_mmd_updates_metric(monkeypatch):
    baseline = np.array([[0.0, 0.0], [2.0, 2.0]])
    mu0 = baseline_mean(baseline)
    new = np.array([[1.0, 1.0], [3.0, 3.0]])
    captured: dict[str, float] = {}
    monkeypatch.setattr(
        drift,
        "update_metric",
        lambda name, value, labels=None: captured.setdefault(name, value),
    )
    mmd2 = embedding_mmd(new, mu0)
    assert captured["embedding_mmd"] == pytest.approx(mmd2)
    assert mmd2 == pytest.approx(2.0)
