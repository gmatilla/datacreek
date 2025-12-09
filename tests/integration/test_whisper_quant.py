"""Tests for quantized Whisper transcription using bitsandbytes."""

import sys
import types

import torch

from datacreek.parsers.whisper_audio_parser import WhisperAudioParser


def test_whisper_parser_int8(monkeypatch):
    """Ensure matmul_8bit hooks into Whisper when running on CPU."""
    calls = {"matmul": 0}

    def fake_transcribe_audio_batch(paths):
        """Return empty results to force fallback."""
        return ["" for _ in paths]

    class FakeModel:
        """Dummy whisper model calling ``torch.matmul``."""

        def transcribe(self, path):
            a = torch.zeros(1, 1)
            torch.matmul(a, a)
            return {"text": "done"}

    def fake_load_model(name):
        """Return the dummy model."""
        return FakeModel()

    def fake_matmul(a, b):
        """Record usage and return zeros."""
        calls["matmul"] += 1
        return torch.zeros(1, 1)

    monkeypatch.setitem(
        sys.modules,
        "bitsandbytes.functional",
        types.SimpleNamespace(matmul_8bit=fake_matmul),
    )
    monkeypatch.setitem(
        sys.modules, "whisper", types.SimpleNamespace(load_model=fake_load_model)
    )
    import importlib

    import datacreek.parsers.whisper_audio_parser as wap

    importlib.reload(wap)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    import datacreek.utils.whisper_batch as wb

    monkeypatch.setattr(wb, "transcribe_audio_batch", fake_transcribe_audio_batch)

    parser = WhisperAudioParser()
    assert parser.parse("x.wav") == "done"
    assert calls["matmul"] == 1
