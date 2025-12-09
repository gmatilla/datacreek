import types

import pytest

import datacreek.utils.audio_vad as audio_vad


class DummyVad:
    def __init__(self, pattern):
        self.pattern = pattern
        self.idx = -1

    def is_speech(self, frame, rate):
        self.idx += 1
        return self.pattern[self.idx]


class DummyModule:
    def __init__(self, pattern):
        self._vad = DummyVad(pattern)

    def Vad(self, *a, **k):
        return self._vad


def test_split_on_silence(monkeypatch):
    pattern = [True, True, False, False, True, True]
    monkeypatch.setattr(audio_vad, "webrtcvad", DummyModule(pattern))
    pcm = b"0" * 120
    segs = audio_vad.split_on_silence(pcm, 1000, frame_ms=10, join_ms=20)
    assert segs == [(0.0, 0.02), (0.04, 0.06)]


def test_requires_webrtcvad(monkeypatch):
    monkeypatch.setattr(audio_vad, "webrtcvad", None)
    with pytest.raises(ImportError):
        audio_vad.split_on_silence(b"0" * 20, 8000)
