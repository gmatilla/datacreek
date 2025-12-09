import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datacreek.utils import audio_vad


def test_split_on_silence(monkeypatch):
    frame_ms = 30
    pattern = [
        True,
        True,
        True,
        True,
        True,  # speech 5 frames
        False,
        False,
        False,
        False,  # short silence 4 frames (<300ms)
        True,
        True,
        True,
        True,
        True,
        True,  # speech 6 frames
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,  # silence 10 frames
        True,
        True,
        True,  # speech 3 frames
        False,
        False,
        False,
        False,  # short silence 4 frames (<300ms)
        True,
        True,
        True,  # speech 3 frames
    ]
    pattern_iter = iter(pattern)

    class FakeVad:
        def __init__(self, mode: int = 3):
            self.mode = mode

        def is_speech(self, frame: bytes, sample_rate: int) -> bool:
            return next(pattern_iter)

    monkeypatch.setattr(audio_vad, "webrtcvad", types.SimpleNamespace(Vad=FakeVad))

    sample_rate = 16000
    frame_len = sample_rate * frame_ms // 1000 * 2
    pcm = b"\x00" * frame_len * len(pattern)
    segments = audio_vad.split_on_silence(pcm, sample_rate, frame_ms=frame_ms)

    assert segments == [(0.0, 0.45), (0.75, 1.05)]
