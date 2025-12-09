import numpy as np
import pytest

from datacreek.analysis.compression import FractalNetPruner
from datacreek.analysis.monitoring import prune_reverts_total


class DummyModel:
    def __init__(self):
        self.w = np.array([0.05, 0.001], dtype=float)

    def named_parameters(self):
        return [("w", self.w)]


def eval_fn(model):
    # simple "perplexity" surrogate as sum of squares
    val = float(np.sum(model.w**2))
    return val


def test_fractalnet_pruner_simple(caplog):
    pruner = FractalNetPruner(lambda_=0.03)
    pruner.model = DummyModel()

    def train_fn(model):
        pass

    caplog.set_level("INFO")
    ok, perp = pruner.prune(eval_fn, train_fn)
    assert ok
    assert perp <= eval_fn(pruner.model) + 1e-9
    assert any("was_reverted=False" in r.message for r in caplog.records)
    assert any("PRUNE_REVERTED=false" in r.message for r in caplog.records)
    if prune_reverts_total is not None:
        assert prune_reverts_total._value.get() == 0


def test_fractalnet_pruner_rollback(tmp_path, caplog):
    pruner = FractalNetPruner(lambda_=0.03)
    pruner.model = DummyModel()

    def bad_train(model):
        model.w[:] = 10.0

    caplog.set_level("INFO")
    ok, perp = pruner.prune(eval_fn, bad_train)
    assert not ok
    # model should have been restored to original weights
    assert np.allclose(pruner.model.w, np.array([0.05, 0.001]))
    assert any("was_reverted=True" in r.message for r in caplog.records)
    assert any("PRUNE_REVERTED=true" in r.message for r in caplog.records)
    if prune_reverts_total is not None:
        assert prune_reverts_total._value.get() >= 1
