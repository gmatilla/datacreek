import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))
import training.trainer_factory as tf


class DummyBase:
    def __init__(self, model, train_dataset, eval_dataset=None, **kwargs):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.kwargs = kwargs


class DummySFT(DummyBase):
    pass


class DummyPPO(DummyBase):
    pass


class DummyDPO(DummyBase):
    pass


def _patch_trainers(monkeypatch):
    monkeypatch.setattr(tf, "SFTTrainer", DummySFT)
    monkeypatch.setattr(tf, "PPOTrainer", DummyPPO)
    monkeypatch.setattr(tf, "DPOTrainer", DummyDPO)
    monkeypatch.setattr(
        tf,
        "TRAINER_MAP",
        {
            "generation": DummySFT,
            "qa": DummySFT,
            "classification": DummySFT,
            "rlhf_ppo": DummyPPO,
            "rlhf_dpo": DummyDPO,
        },
    )


def test_build_trainer_returns_expected_class(monkeypatch):
    _patch_trainers(monkeypatch)
    model = object()
    data = object()

    gen = tf.build_trainer("generation", model, data, batch_size=1)
    assert isinstance(gen, DummySFT)
    assert gen.kwargs["batch_size"] == 1

    qa = tf.build_trainer("qa", model, data)
    assert isinstance(qa, DummySFT)

    cls = tf.build_trainer("classification", model, data)
    assert isinstance(cls, DummySFT)

    ppo = tf.build_trainer("rlhf_ppo", model, data)
    assert isinstance(ppo, DummyPPO)

    dpo = tf.build_trainer("rlhf_dpo", model, data)
    assert isinstance(dpo, DummyDPO)

    with pytest.raises(ValueError):
        tf.build_trainer("unknown", model, data)


def test_build_trainer_raises_when_trl_missing(monkeypatch):
    monkeypatch.setattr(tf, "TRAINER_MAP", {"generation": None})
    with pytest.raises(ImportError):
        tf.build_trainer("generation", object(), object())
