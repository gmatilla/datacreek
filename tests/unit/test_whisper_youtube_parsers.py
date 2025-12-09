import builtins
import sys
import types

import pytest

from datacreek.parsers.whisper_audio_parser import WhisperAudioParser, matmul_8bit
from datacreek.parsers.youtube_parser import YouTubeParser


class DummyModel:
    def __init__(self):
        self.transcribed = []

    def transcribe(self, fp):
        self.transcribed.append(fp)
        return {"text": "hi"}


def test_whisper_parser_batch(monkeypatch, tmp_path):
    """Parser should chunk the audio and concatenate transcripts."""

    called = {}

    def fake_batch(paths):
        called["paths"] = list(paths)
        return [f"t{i}" for i in range(len(paths))]

    monkeypatch.setitem(
        sys.modules,
        "datacreek.utils.whisper_batch",
        types.SimpleNamespace(transcribe_audio_batch=fake_batch),
    )

    monkeypatch.setitem(
        sys.modules,
        "datacreek.utils.audio_vad",
        types.SimpleNamespace(
            split_on_silence=lambda pcm, rate, join_ms=300: [(0.0, 0.5), (0.5, 1.0)]
        ),
    )

    class WaveRead:
        def __init__(self, *a, **k):
            self.r = 16000

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def readframes(self, n):
            return b"0" * 2 * n

        def getnframes(self):
            return 16000

        def getframerate(self):
            return self.r

        def getnchannels(self):
            return 1

        def getsampwidth(self):
            return 2

    class WaveWrite(WaveRead):
        def setnchannels(self, c):
            pass

        def setsampwidth(self, w):
            pass

        def setframerate(self, r):
            pass

        def writeframes(self, data):
            pass

    def fake_wave(path, mode="rb"):
        return WaveWrite()

    monkeypatch.setitem(sys.modules, "wave", types.SimpleNamespace(open=fake_wave))

    parser = WhisperAudioParser()
    f = tmp_path / "a.wav"
    f.write_bytes(b"0")
    out = parser.parse(str(f))

    assert out == "t0 t1"
    assert len(called["paths"]) == 2


def test_whisper_parser_whisper(monkeypatch, tmp_path):
    """Fallback to the ``whisper`` package should handle segmented audio."""

    monkeypatch.setitem(
        sys.modules,
        "datacreek.utils.whisper_batch",
        types.SimpleNamespace(transcribe_audio_batch=None),
    )
    fake_model = DummyModel()
    monkeypatch.setitem(
        sys.modules,
        "whisper",
        types.SimpleNamespace(load_model=lambda name: fake_model),
    )

    class FakeTorch:
        def __init__(self):
            self.cuda = types.SimpleNamespace(is_available=lambda: False)
            self.matmul = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setattr(
        "datacreek.parsers.whisper_audio_parser.matmul_8bit", None, raising=False
    )
    monkeypatch.setitem(
        sys.modules,
        "datacreek.utils.audio_vad",
        types.SimpleNamespace(
            split_on_silence=lambda pcm, rate, join_ms=300: [(0.0, 1.0)]
        ),
    )

    class WaveStub:
        def __init__(self, *a, **k):
            self.r = 16000

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def readframes(self, n):
            return b"0" * 2 * n

        def getnframes(self):
            return 16000

        def getframerate(self):
            return self.r

        def getnchannels(self):
            return 1

        def getsampwidth(self):
            return 2

    monkeypatch.setitem(
        sys.modules, "wave", types.SimpleNamespace(open=lambda p, mode="rb": WaveStub())
    )
    parser = WhisperAudioParser()
    f = tmp_path / "a.wav"
    f.write_bytes(b"0")
    assert parser.parse(str(f)) == "hi"
    assert fake_model.transcribed


def test_youtube_parser(monkeypatch):
    class FakeYT:
        def __init__(self, url):
            self.video_id = "abc"
            self.title = "T"
            self.author = "A"
            self.length = 1

    monkeypatch.setitem(sys.modules, "pytubefix", types.SimpleNamespace(YouTube=FakeYT))
    monkeypatch.setitem(
        sys.modules,
        "youtube_transcript_api",
        types.SimpleNamespace(
            YouTubeTranscriptApi=types.SimpleNamespace(
                get_transcript=lambda vid: [{"text": "foo"}, {"text": "bar"}]
            )
        ),
    )
    parser = YouTubeParser()
    text = parser.parse("http://youtube.com/watch?v=abc")
    assert "Title: T" in text
    assert "foo" in text and "bar" in text
