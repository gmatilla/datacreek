import sys
import types

import numpy as np
import pytest

from datacreek.analysis import compression


def test_save_and_restore_checkpoint(tmp_path):
    obj = {"x": 1}
    path = tmp_path / "model.pkl"
    compression.save_checkpoint(str(path), obj)
    restored = compression.restore_checkpoint(str(path))
    assert restored == obj


def test_prune_fractalnet_numpy():
    w = np.array([1.0, 0.5, -0.1, 0.2])
    pruned = compression.prune_fractalnet(w, ratio=0.5)
    assert np.allclose(pruned, np.array([1.0, 0.5, 0.0, 0.0]))
    zeros = compression.prune_fractalnet(w, ratio=0.0)
    assert np.allclose(zeros, np.zeros_like(w))


def test_prune_fractalnet_without_numpy(monkeypatch):
    monkeypatch.setattr(compression, "np", None)
    result = compression.prune_fractalnet([1.0, -2.0, 0.5], ratio=1 / 3)
    assert result == [0.0, -2.0, 0.0]


class DummyCounter:
    def __init__(self):
        self.count = 0

    def inc(self):
        self.count += 1


class DummyModel:
    def __init__(self, weights):
        self.weight = np.array(weights, dtype=float)


def test_fractalnetpruner_prune_no_revert(monkeypatch, tmp_path):
    model = DummyModel([0.1, 0.03, 0.07])
    pruner = compression.FractalNetPruner(lambda_=0.05)
    pruner.model = model

    calls = []

    def fake_save(path, obj):
        calls.append(path)

    monkeypatch.setattr(compression, "save_checkpoint", fake_save)
    monkeypatch.setattr(compression, "restore_checkpoint", lambda p: None)
    monkeypatch.setattr(
        compression, "prune_reverts_total", DummyCounter(), raising=False
    )
    fake_cfg_mod = types.SimpleNamespace(
        load_config=lambda: {"compression": {"magnitude": 0.05}}
    )
    monkeypatch.setitem(sys.modules, "datacreek.utils.config", fake_cfg_mod)

    vals = iter([1.0, 0.995])
    pruned_ok, ppl = pruner.prune(lambda m: next(vals))

    assert pruned_ok is True
    assert ppl == pytest.approx(0.995)
    assert model.weight[1] == 0.0
    assert "pruned.ok" in calls


def test_fractalnetpruner_prune_revert(monkeypatch):
    model = DummyModel([0.1, 0.2])
    pruner = compression.FractalNetPruner(lambda_=0.05)
    pruner.model = model

    dummy_counter = DummyCounter()
    monkeypatch.setattr(
        compression, "prune_reverts_total", dummy_counter, raising=False
    )
    monkeypatch.setattr(compression, "save_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(compression, "restore_checkpoint", lambda p: "restored")
    fake_cfg_mod = types.SimpleNamespace(
        load_config=lambda: {"compression": {"magnitude": 0.05}}
    )
    monkeypatch.setitem(sys.modules, "datacreek.utils.config", fake_cfg_mod)

    vals = iter([1.0, 2.0])
    accepted, ppl = pruner.prune(lambda m: next(vals))

    assert accepted is False
    assert ppl == 2.0
    assert pruner.model == "restored"
    assert dummy_counter.count == 1


def test_fractalnetpruner_load_failure(monkeypatch):
    pruner = compression.FractalNetPruner()
    monkeypatch.setattr(pruner, "load", lambda: None)
    with pytest.raises(RuntimeError):
        pruner.prune(lambda m: 0.0)
