import builtins
import sys
import types

import pytest

from datacreek.parsers.image_parser import ImageParser


class DummyEl:
    def __init__(self, text):
        self.text = text


def test_image_parser_success(monkeypatch):
    def fake_partition_image(filename):
        return [DummyEl("foo"), DummyEl("bar")]

    monkeypatch.setitem(
        sys.modules,
        "unstructured.partition.image",
        types.SimpleNamespace(partition_image=fake_partition_image),
    )
    parser = ImageParser()
    assert parser.parse("img.png") == "foo\nbar"


def test_image_parser_missing_dep(monkeypatch):
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "unstructured.partition.image":
            raise ImportError("nope")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    parser = ImageParser()
    with pytest.raises(ImportError):
        parser.parse("img.png")
    monkeypatch.setattr(builtins, "__import__", orig_import)
