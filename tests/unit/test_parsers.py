import builtins
import importlib
import os
import sys
import tempfile
import types

import pytest

from datacreek.parsers.audio_parser import AudioParser
from datacreek.parsers.base import BaseParser
from datacreek.parsers.code_parser import CodeParser


class DummyParser(BaseParser):
    def parse(self, file_path: str) -> str:
        return f"parsed:{file_path}"


def test_base_parser_errors():
    bp = BaseParser()
    with pytest.raises(NotImplementedError):
        bp.parse("foo")
    with pytest.raises(RuntimeError):
        bp.save("txt", "out")


def test_code_parser(tmp_path):
    p = CodeParser()
    file = tmp_path / "code.py"
    file.write_text("print('hi')")
    assert p.parse(str(file)) == "print('hi')"


def test_audio_parser_missing_dep(monkeypatch):
    monkeypatch.setitem(sys.modules, "speech_recognition", None)
    parser = AudioParser()
    with pytest.raises(ImportError):
        parser.parse("x.wav")


def test_audio_parser_success(monkeypatch, tmp_path):
    class DummyRec:
        class UnknownValueError(Exception):
            pass

        class RequestError(Exception):
            pass

        class Recognizer:
            def record(self, source):
                return b""

            def recognize_sphinx(self, audio):
                return "ok"

        class AudioFile:
            def __init__(self, path):
                pass

            def __enter__(self):
                return object()

            def __exit__(self, exc_type, exc, tb):
                pass

    monkeypatch.setitem(sys.modules, "speech_recognition", DummyRec)
    parser = AudioParser()
    assert parser.parse(str(tmp_path / "f.wav")) == "ok"
