import sys
import types

import pytest

from datacreek.parsers.ppt_parser import PPTParser


class DummyEl:
    def __init__(self, text: str | None):
        self.text = text


def test_ppt_parser_text(monkeypatch):
    def fake_partition_pptx(filename):
        return [DummyEl("a"), DummyEl("b")]

    monkeypatch.setitem(
        sys.modules,
        "unstructured.partition.pptx",
        types.SimpleNamespace(partition_pptx=fake_partition_pptx),
    )
    parser = PPTParser()
    assert parser.parse("f.pptx") == "a\nb"


def test_ppt_parser_return_elements(monkeypatch):
    elems = [DummyEl("x")]

    def fake_partition_pptx(filename):
        return elems

    monkeypatch.setitem(
        sys.modules,
        "unstructured.partition.pptx",
        types.SimpleNamespace(partition_pptx=fake_partition_pptx),
    )
    parser = PPTParser()
    assert parser.parse("f.pptx", return_elements=True) is elems


def test_ppt_parser_error(monkeypatch):
    def fake_partition_pptx(filename):
        raise ValueError("boom")

    monkeypatch.setitem(
        sys.modules,
        "unstructured.partition.pptx",
        types.SimpleNamespace(partition_pptx=fake_partition_pptx),
    )
    parser = PPTParser()
    with pytest.raises(RuntimeError):
        parser.parse("f.pptx")
