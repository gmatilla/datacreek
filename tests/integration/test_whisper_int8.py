"""Tests for Whisper int8 GEMM acceleration on CPU."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def test_whisper_int8_cpu(monkeypatch):
    """``transcribe_audio_batch`` should use 8-bit matmul on CPU."""
    calls = {"matmul": 0}

    def fake_matmul(a, b):
        calls["matmul"] += 1
        return 0

    dummy_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        matmul=lambda a, b: 0,
        zeros=lambda *a, **k: 0,
    )

    monkeypatch.setitem(sys.modules, "torch", dummy_torch)
    bnb_mod = types.ModuleType("bitsandbytes")
    bnb_mod.functional = types.SimpleNamespace(matmul_8bit=fake_matmul)
    monkeypatch.setitem(sys.modules, "bitsandbytes", bnb_mod)
    monkeypatch.setitem(sys.modules, "bitsandbytes.functional", bnb_mod.functional)

    spec = importlib.util.spec_from_file_location(
        "whisper_batch",
        Path(__file__).resolve().parents[1]
        / "datacreek"
        / "utils"
        / "whisper_batch.py",
    )
    wb = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(wb)
    wb.torch = dummy_torch

    class DummyModel:
        def transcribe(self, path: str, max_length: int = 30) -> str:
            assert wb.torch.matmul is fake_matmul
            wb.torch.matmul(0, 0)
            return "ok"

    monkeypatch.setattr(wb, "_get_model", lambda *a, **k: DummyModel())

    gauge_vals: dict[str, float] = {}

    class DummyGauge:
        def labels(self, **kw):
            gauge_vals.update(kw)
            return self

        def set(self, val: float) -> None:
            gauge_vals["xrt"] = val

    gauge = DummyGauge()
    mon_stub = types.SimpleNamespace(
        whisper_xrt=gauge,
        _METRICS={"whisper_xrt": gauge},
        update_metric=lambda name, val, labels=None: gauge.labels(**(labels or {})).set(
            val
        ),
        whisper_fallback_total=None,
    )
    sys.modules["datacreek.analysis.monitoring"] = mon_stub

    wb.transcribe_audio_batch(["a.wav"], device="cpu")
    sys.modules.pop("datacreek.analysis.monitoring", None)

    assert calls["matmul"] >= 1
    assert gauge_vals.get("device") == "cpu"
    assert gauge_vals.get("xrt", 0.0) <= 1.5
