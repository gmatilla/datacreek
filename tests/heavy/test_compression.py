import sys
import types

import numpy as np
import pytest

from datacreek.analysis import compression


def test_save_and_restore(tmp_path):
    path = tmp_path / "model.pkl"
    obj = {"a": 1}
    compression.save_checkpoint(str(path), obj)
    restored = compression.restore_checkpoint(str(path))
    assert restored == obj


def test_prune_fractalnet_numpy():
    w = np.array([1.0, 0.5, -0.1, 0.2])
    result = compression.prune_fractalnet(w, ratio=0.5)
    assert np.count_nonzero(result) == 2


def test_prune_fractalnet_without_numpy(monkeypatch):
    monkeypatch.setattr(compression, "np", None)
    result = compression.prune_fractalnet([1.0, -2.0, 0.5], ratio=1 / 3)
    assert result == [0.0, -2.0, 0.0]

    zeroed = compression.prune_fractalnet([1.0, -2.0], ratio=0.0)
    assert zeroed == [0.0, 0.0]


class DummyCounter:
    def __init__(self):
        self.count = 0

    def inc(self):
        self.count += 1


class DummyModel:
    def __init__(self, weights):
        self.weight = np.array(weights, dtype=float)

    def modules(self):
        yield types.SimpleNamespace(weight=self.weight)


def test_fractalnetpruner_prune_no_revert(monkeypatch):
    model = DummyModel([0.1, 0.03, 0.07])
    pruner = compression.FractalNetPruner(lambda_=0.05)
    pruner.model = model

    monkeypatch.setattr(
        compression, "prune_reverts_total", DummyCounter(), raising=False
    )
    monkeypatch.setattr(compression, "save_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(compression, "restore_checkpoint", lambda p: None)
    fake_cfg = types.SimpleNamespace(
        load_config=lambda: {"compression": {"magnitude": 0.05}}
    )
    monkeypatch.setitem(sys.modules, "datacreek.utils.config", fake_cfg)

    vals = iter([1.0, 0.995])
    accepted, ppl = pruner.prune(lambda m: next(vals))

    assert accepted is True
    assert ppl == pytest.approx(0.995)
    assert model.weight[1] == 0.0


def test_fractalnetpruner_prune_revert(monkeypatch):
    model = DummyModel([0.1, 0.2])
    pruner = compression.FractalNetPruner(lambda_=0.05)
    pruner.model = model

    counter = DummyCounter()
    monkeypatch.setattr(compression, "prune_reverts_total", counter, raising=False)
    monkeypatch.setattr(compression, "save_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(compression, "restore_checkpoint", lambda p: "restored")
    fake_cfg = types.SimpleNamespace(
        load_config=lambda: {"compression": {"magnitude": 0.05}}
    )
    monkeypatch.setitem(sys.modules, "datacreek.utils.config", fake_cfg)

    vals = iter([1.0, 1.1])
    accepted, ppl = pruner.prune(lambda m: next(vals))

    assert accepted is False
    assert ppl == 1.1
    assert pruner.model == "restored"
    assert counter.count == 1
