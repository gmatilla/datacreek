"""Tests for :mod:`training.unsloth_loader`.

The project uses a hybrid layout where helper modules like ``training`` are
not installed as packages. To ensure they are importable during unit tests we
manually append the repository root to ``sys.path`` before performing the
imports below.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Make repository root importable so ``training`` can be resolved.
sys.path.append(str(Path(__file__).resolve().parents[2]))

from training.unsloth_loader import add_lora, load_model


class DummyFLM:
    """Minimal stand-in for :class:`unsloth.FastLanguageModel`."""

    def __init__(self) -> None:
        self.from_pretrained_called = False
        self.get_peft_model_called = False
        self.args = SimpleNamespace()

    @classmethod
    def from_pretrained(
        cls, model_id, **kwargs
    ):  # noqa: D401 - signature mirrors unsloth
        """Mocked loader capturing the provided arguments."""
        instance = cls()
        instance.from_pretrained_called = True
        instance.args.model_id = model_id
        instance.args.kwargs = kwargs
        return instance

    @staticmethod
    def get_peft_model(model, **kwargs):  # noqa: D401 - signature mirrors unsloth
        """Mocked PEFT wrapper capturing arguments."""
        model.get_peft_model_called = True
        model.args.peft_kwargs = kwargs
        return model


@pytest.fixture(autouse=True)
def patch_fast_language_model(monkeypatch):
    dummy = DummyFLM
    monkeypatch.setattr("training.unsloth_loader.FastLanguageModel", dummy)
    yield


def test_load_model_uses_from_pretrained():
    model = load_model("tiny", bits=4, max_seq=128, foo="bar")
    assert model.from_pretrained_called
    assert model.args.model_id == "tiny"
    assert model.args.kwargs["foo"] == "bar"
    assert model.args.kwargs["load_in_4bit"] is True
    assert model.args.kwargs["max_seq_len"] == 128


def test_add_lora_uses_get_peft_model():
    model = DummyFLM()
    wrapped = add_lora(model, r=8, alpha=16, target_modules=["q_proj"])
    assert wrapped.get_peft_model_called
    assert wrapped.args.peft_kwargs["r"] == 8
    assert wrapped.args.peft_kwargs["lora_alpha"] == 16
    assert wrapped.args.peft_kwargs["target_modules"] == ["q_proj"]
