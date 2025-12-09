import os
import sys
import types

sys.modules.setdefault("transformers", types.ModuleType("transformers"))
setattr(sys.modules["transformers"], "pipeline", lambda *a, **k: None)

from datacreek.analysis.ingestion import (
    blip_caption_image,
    partition_files_to_atoms,
    transcribe_audio,
)
from datacreek.utils import image_captioning, whisper_batch


def test_partition_files_to_atoms(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text("a\nb\nc")
    atoms = partition_files_to_atoms(str(p))
    assert atoms == ["a", "b", "c"]


def test_transcribe_audio_fallback(tmp_path):
    p = tmp_path / "audio.wav"
    p.write_bytes(b"fake")
    assert transcribe_audio(str(p)) == ""


def test_blip_caption_image_fallback(tmp_path):
    p = tmp_path / "img.jpg"
    p.write_bytes(b"fake")
    assert blip_caption_image(str(p)) == ""


def test_parse_code_to_atoms(tmp_path):
    code = """\
 def foo(x):
     return x+1
 
 class Bar:
     def baz(self):
         pass
 """
    p = tmp_path / "sample.py"
    p.write_text(code)
    from datacreek.analysis.ingestion import parse_code_to_atoms

    atoms = parse_code_to_atoms(str(p))
    assert len(atoms) == 2
    assert "def foo" in atoms[0]
    assert "class Bar" in atoms[1]


def test_caption_images_parallel(monkeypatch):
    paths = [f"img{i}.png" for i in range(300)]
    calls: list[str] = []

    def _fake(path: str) -> str:
        calls.append(path)
        return f"cap-{path}"

    monkeypatch.setattr(image_captioning, "caption_image", _fake)

    captions = image_captioning.caption_images_parallel(
        paths, max_workers=4, chunk_size=128
    )
    assert captions == [f"cap-{p}" for p in paths]
    assert calls == paths


def test_transcribe_audio_batch(monkeypatch):
    paths = [f"a{i}.wav" for i in range(5)]
    calls: list[str] = []

    class DummyModel:
        def transcribe(self, path: str, max_length: int = 30) -> str:
            calls.append(path)
            return f"txt-{path}"

    monkeypatch.setattr(whisper_batch, "_get_model", lambda *a, **k: DummyModel())

    result = whisper_batch.transcribe_audio_batch(paths, batch_size=2)
    assert result == [f"txt-{p}" for p in paths]
    assert calls == paths


def test_whisper_batch_gpu_fallback(monkeypatch):
    paths = ["a.wav", "b.wav"]
    calls: list[tuple[str, str]] = []

    class GPUModel:
        def transcribe(self, path: str, max_length: int = 30) -> str:
            raise RuntimeError("CUDA out of memory")

    class CPUModel:
        def transcribe(self, path: str, max_length: int = 30) -> str:
            calls.append(("cpu", path))
            return f"cpu-{path}"

    def fake_get(model="tiny.en", fp16=True, device=None, int8=False):
        return GPUModel() if device == "cuda" else CPUModel()

    fake_get.cache_clear = lambda: None

    monkeypatch.setattr(whisper_batch, "_get_model", fake_get)
    monkeypatch.setattr(
        whisper_batch,
        "torch",
        type(
            "T",
            (),
            {"cuda": type("C", (), {"is_available": staticmethod(lambda: True)})},
        )(),
    )

    fb = {"n": 0}

    class DummyCounter:
        def inc(self):
            fb["n"] += 1

    vals = []

    class DummyGauge:
        def labels(self, **kwargs):
            return self

        def set(self, v: float):
            vals.append(v)

    monkeypatch.setattr(
        "datacreek.analysis.monitoring.whisper_fallback_total",
        DummyCounter(),
        raising=False,
    )
    g = DummyGauge()
    monkeypatch.setattr(
        "datacreek.analysis.monitoring.whisper_xrt",
        g,
        raising=False,
    )
    import datacreek.analysis.monitoring as mon

    monkeypatch.setitem(mon._METRICS, "whisper_xrt", g)

    result = whisper_batch.transcribe_audio_batch(paths, batch_size=2)
    assert result == ["cpu-a.wav", "cpu-b.wav"]
    assert fb["n"] == 1
    assert calls == [("cpu", "a.wav"), ("cpu", "b.wav")]
    assert len(vals) == 1


def test_whisper_parser_uses_batch(monkeypatch):
    captured: list[str] = []

    def fake_batch(paths, **kw):
        captured.extend(paths)
        return ["hello"]

    monkeypatch.setattr(whisper_batch, "transcribe_audio_batch", fake_batch)
    from datacreek.parsers.whisper_audio_parser import WhisperAudioParser

    parser = WhisperAudioParser()
    assert parser.parse("x.wav") == "hello"
    assert captured == ["x.wav"]
