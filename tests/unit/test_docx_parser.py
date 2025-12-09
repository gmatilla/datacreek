import sys
import types

import pytest

from datacreek.parsers.docx_parser import DOCXParser


class DummyElement:
    def __init__(self, text: str | None):
        self.text = text


def test_docx_parser_returns_joined_text(monkeypatch):
    calls = {}

    def fake_partition_docx(filename):
        calls["filename"] = filename
        return [DummyElement("foo"), DummyElement("bar")]

    monkeypatch.setitem(
        sys.modules,
        "unstructured.partition.docx",
        types.SimpleNamespace(partition_docx=fake_partition_docx),
    )
    parser = DOCXParser()
    result = parser.parse("dummy.docx")
    assert result == "foo\nbar"
    assert calls["filename"] == "dummy.docx"


def test_docx_parser_return_elements(monkeypatch):
    elems = [DummyElement("a")]

    def fake_partition_docx(filename):
        return elems

    monkeypatch.setitem(
        sys.modules,
        "unstructured.partition.docx",
        types.SimpleNamespace(partition_docx=fake_partition_docx),
    )
    parser = DOCXParser()
    result = parser.parse("x.docx", return_elements=True)
    assert result is elems


def test_docx_parser_error(monkeypatch):
    def fake_partition_docx(filename):
        raise ValueError("boom")

    monkeypatch.setitem(
        sys.modules,
        "unstructured.partition.docx",
        types.SimpleNamespace(partition_docx=fake_partition_docx),
    )
    parser = DOCXParser()
    with pytest.raises(RuntimeError):
        parser.parse("x.docx")
