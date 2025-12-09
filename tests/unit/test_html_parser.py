import sys
import types

import pytest

from datacreek.parsers.html_parser import HTMLParser


class DummyEl:
    def __init__(self, text: str | None):
        self.text = text


def test_html_parser_file(monkeypatch):
    captured = {}

    def fake_partition_html(url=None, filename=None):
        captured["url"] = url
        captured["filename"] = filename
        return [DummyEl("foo"), DummyEl("bar")]

    monkeypatch.setitem(
        sys.modules,
        "unstructured.partition.html",
        types.SimpleNamespace(partition_html=fake_partition_html),
    )
    parser = HTMLParser()
    result = parser.parse("local.html")
    assert result == "foo\nbar"
    assert captured["url"] is None
    assert captured["filename"] == "local.html"


def test_html_parser_url(monkeypatch):
    captured = {}

    def fake_partition_html(url=None, filename=None):
        captured["url"] = url
        captured["filename"] = filename
        return [DummyEl("x")]

    monkeypatch.setitem(
        sys.modules,
        "unstructured.partition.html",
        types.SimpleNamespace(partition_html=fake_partition_html),
    )
    parser = HTMLParser()
    result = parser.parse("https://foo.bar")
    assert result == "x"
    assert captured["url"] == "https://foo.bar"
    assert captured["filename"] is None


def test_html_parser_return_elements(monkeypatch):
    elems = [DummyEl("hey")]

    def fake_partition_html(url=None, filename=None):
        return elems

    monkeypatch.setitem(
        sys.modules,
        "unstructured.partition.html",
        types.SimpleNamespace(partition_html=fake_partition_html),
    )
    parser = HTMLParser()
    assert parser.parse("any.html", return_elements=True) is elems


def test_html_parser_error(monkeypatch):
    def fake_partition_html(**_):
        raise ValueError("boom")

    monkeypatch.setitem(
        sys.modules,
        "unstructured.partition.html",
        types.SimpleNamespace(partition_html=fake_partition_html),
    )
    parser = HTMLParser()
    with pytest.raises(RuntimeError):
        parser.parse("fail.html")
