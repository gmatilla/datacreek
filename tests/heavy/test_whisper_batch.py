import importlib
import types

import pytest

import datacreek.utils.whisper_batch as wb


class DummyModel:
    def __init__(self, *args, **kwargs):
        self.calls = []
        self.kwargs = kwargs

    def transcribe(self, path, max_length=0):
        self.calls.append(path)
        return f"text:{path}"


def type_error_model(*args, **kwargs):
    if "compute_type" in kwargs:
        raise TypeError("old api")
    return DummyModel(*args, **kwargs)


def test_get_model_old_api(monkeypatch):
    monkeypatch.setattr(wb, "Whisper", type_error_model)
    wb._get_model.cache_clear()
    model = wb._get_model("model", fp16=False, int8=True)
    assert isinstance(model, DummyModel)
    assert "compute_type" not in model.kwargs
    assert model.kwargs.get("fp16") is False


def test_transcribe_audio_batch_cpu(monkeypatch):
    dummy = DummyModel()
    monkeypatch.setattr(wb, "Whisper", lambda *a, **k: dummy)
    torch_stub = types.SimpleNamespace(
        matmul=lambda a, b: (a, b),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    bnb_stub = types.SimpleNamespace(matmul_8bit=lambda a, b: (b, a))
    monkeypatch.setattr(wb, "torch", torch_stub)
    monkeypatch.setattr(wb, "bnb_fn", bnb_stub)
    metrics = {}

    def update_metric(name, value, labels):
        metrics[name] = value

    mod = importlib.import_module("datacreek.analysis.monitoring")
    monkeypatch.setattr(mod, "update_metric", update_metric)
    transcripts = wb.transcribe_audio_batch(
        ["a", "b"], device="cpu", batch_size=4, max_seconds=1
    )
    assert transcripts == ["text:a", "text:b"]
    assert metrics["whisper_xrt"] > 0
    assert dummy.calls == ["a", "b"]
