import sys
import types

import pytest

# Provide stub modules to avoid heavy imports
sys.modules.setdefault("transformers", types.ModuleType("transformers"))


# Stub monitoring module to avoid importing the whole package
def _update_metric(name: str, value: float, labels=None):
    g = mon_stub._METRICS.get(name)
    if g is not None:
        if labels:
            g.labels(**labels).set(value)
        else:
            g.set(value)


mon_stub = types.SimpleNamespace(
    whisper_xrt=None,
    _METRICS={"whisper_xrt": None},
    update_metric=_update_metric,
    whisper_fallback_total=None,
)
# Stub datacreek package hierarchy so imports in whisper_batch resolve
dc_stub = types.ModuleType("datacreek")
analysis_stub = types.ModuleType("datacreek.analysis")
analysis_stub.monitoring = mon_stub
dc_stub.analysis = analysis_stub
sys.modules.setdefault("datacreek", dc_stub)
sys.modules.setdefault("datacreek.analysis", analysis_stub)
sys.modules.setdefault("datacreek.analysis.monitoring", mon_stub)

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "whisper_batch",
    Path(__file__).resolve().parents[1] / "datacreek" / "utils" / "whisper_batch.py",
)
whisper_batch = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(whisper_batch)
# remove stub modules so other tests can import real package
sys.modules.pop("datacreek.analysis.monitoring", None)
sys.modules.pop("datacreek.analysis", None)
sys.modules.pop("datacreek", None)


def test_cpu_route(monkeypatch):
    class DummyModel:
        def __init__(self):
            self.calls = []

        def transcribe(self, path: str, max_length: int = 30) -> str:
            self.calls.append(path)
            return "ok"

    monkeypatch.setattr(whisper_batch, "_get_model", lambda *a, **k: DummyModel())
    monkeypatch.setattr(
        whisper_batch,
        "torch",
        type(
            "T",
            (),
            {"cuda": type("C", (), {"is_available": staticmethod(lambda: False)})},
        )(),
    )

    vals = {}

    class DummyGauge:
        def labels(self, **kwargs):
            vals.update(kwargs)
            return self

        def set(self, v: float):
            vals["xrt"] = v

    gauge = DummyGauge()
    mon_stub.whisper_xrt = gauge
    mon_stub._METRICS["whisper_xrt"] = gauge
    sys.modules["datacreek.analysis.monitoring"] = mon_stub

    result = whisper_batch.transcribe_audio_batch(["a.wav"], batch_size=4)
    sys.modules.pop("datacreek.analysis.monitoring", None)
    assert result == ["ok"]
    assert vals.get("device") == "cpu"
    assert vals["xrt"] <= 1.5


@pytest.mark.gpu
def test_gpu_route(monkeypatch):
    class DummyModel:
        def __init__(self):
            self.calls = []

        def transcribe(self, path: str, max_length: int = 30) -> str:
            self.calls.append(path)
            return "ok"

    monkeypatch.setattr(whisper_batch, "_get_model", lambda *a, **k: DummyModel())
    monkeypatch.setattr(
        whisper_batch,
        "torch",
        type(
            "T",
            (),
            {"cuda": type("C", (), {"is_available": staticmethod(lambda: True)})},
        )(),
    )

    ticks = [0.0, 0.1]
    monkeypatch.setattr(whisper_batch.time, "perf_counter", lambda: ticks.pop(0))

    vals = {}

    class DummyGauge:
        def labels(self, **kwargs):
            vals.update(kwargs)
            return self

        def set(self, v: float):
            vals["xrt"] = v

    gauge = DummyGauge()
    mon_stub.whisper_xrt = gauge
    mon_stub._METRICS["whisper_xrt"] = gauge
    sys.modules["datacreek.analysis.monitoring"] = mon_stub

    result = whisper_batch.transcribe_audio_batch(["a.wav"], batch_size=4)
    sys.modules.pop("datacreek.analysis.monitoring", None)
    assert result == ["ok"]
    assert vals.get("device") == "cuda"
    assert vals["xrt"] <= 0.5
