"""Tests for Whisper realtime factor benchmark."""

import importlib.abc
import importlib.util
import sys
import types
from pathlib import Path


def test_bench_whisper_xrt(tmp_path, monkeypatch):
    """Run the benchmark script with a dummy parser to verify output."""
    spec = importlib.util.spec_from_file_location(
        "bench_whisper_xrt",
        Path(__file__).resolve().parents[1] / "scripts" / "bench_whisper_xrt.py",
    )
    bench = importlib.util.module_from_spec(spec)
    assert isinstance(spec.loader, importlib.abc.Loader)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(
        sys.modules,
        "bitsandbytes.functional",
        types.SimpleNamespace(matmul_8bit=lambda a, b: None),
    )
    spec.loader.exec_module(bench)

    class DummyParser:
        """Simple parser counting invocations."""

        def __init__(self) -> None:
            self.calls = 0

        def parse(self, path: str) -> str:
            self.calls += 1
            return "ok"

    monkeypatch.setattr(bench, "WhisperAudioParser", DummyParser)
    ticks = [0.0, 0.05]
    monkeypatch.setattr(bench.time, "perf_counter", lambda: ticks.pop(0))
    out = tmp_path / "metrics"
    res = bench.main(
        [
            "--file",
            "x.wav",
            "--loops",
            "1",
            "--max-seconds",
            "1",
            "--output",
            str(out),
        ]
    )
    assert res["whisper_xrt"] <= 1.5  # noqa: S101
    assert "whisper_xrt" in out.read_text()  # noqa: S101
