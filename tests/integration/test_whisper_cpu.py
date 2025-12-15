import sys
import types
import unittest.mock
import importlib 
import pytest

# Stub monitoring module to capture metrics
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

@pytest.fixture(scope="module")
def whisper_module():
    """Load whisper_batch with mocked dependencies."""
    # Stub transformers and datacreek monitoring
    mock_modules = {
        "transformers": types.ModuleType("transformers"),
        "datacreek.analysis.monitoring": mon_stub,
    }
    
    with unittest.mock.patch.dict(sys.modules, mock_modules):
        from datacreek.utils import whisper_batch
        importlib.reload(whisper_batch)
        yield whisper_batch





def test_cpu_route(monkeypatch, whisper_module):
    class DummyModel:
        def __init__(self):
            self.calls = []

        def transcribe(self, path: str, max_length: int = 30) -> str:
            self.calls.append(path)
            return "ok"

    monkeypatch.setattr(whisper_module, "_get_model", lambda *a, **k: DummyModel())
    # Mock torch here since it is imported inside the module or used from it
    monkeypatch.setattr(
        whisper_module,
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
    
    # Ensure monitoring in sys.modules is consistent for the test execution if needed,
    # but the module already holds reference to mon_stub via import.
    
    result = whisper_module.transcribe_audio_batch(["a.wav"], batch_size=4)
    # No need to pop, handled by cleanup/fixture usually, but here we modify valid dict
    
    assert result == ["ok"]
    assert vals.get("device") == "cpu"
    assert vals["xrt"] <= 1.5


@pytest.mark.gpu
def test_gpu_route(monkeypatch, whisper_module):
    class DummyModel:
        def __init__(self):
            self.calls = []

        def transcribe(self, path: str, max_length: int = 30) -> str:
            self.calls.append(path)
            return "ok"

    monkeypatch.setattr(whisper_module, "_get_model", lambda *a, **k: DummyModel())
    monkeypatch.setattr(
        whisper_module,
        "torch",
        type(
            "T",
            (),
            {"cuda": type("C", (), {"is_available": staticmethod(lambda: True)})},
        )(),
    )

    ticks = [0.0, 0.1]
    monkeypatch.setattr(whisper_module.time, "perf_counter", lambda: ticks.pop(0))

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

    result = whisper_module.transcribe_audio_batch(["a.wav"], batch_size=4)
    assert result == ["ok"]
    assert vals.get("device") == "cuda"
    assert vals["xrt"] <= 0.5

