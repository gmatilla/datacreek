import builtins
import os
import types

import pytest

import datacreek.utils.text as text


def test_split_into_chunks_basic():
    data = "a\n\nb\n\nc"
    # short chunk_size causes paragraphs to be merged
    assert text.split_into_chunks(data, chunk_size=2) == ["a\n\nb", "c"]


def test_split_methods(monkeypatch):
    monkeypatch.setattr(
        text, "sliding_window_chunks", lambda *a, **k: ["sl"], raising=False
    )
    assert text.split_into_chunks("x", method="sliding") == ["sl"]
    monkeypatch.setattr(
        text, "semantic_chunk_split", lambda *a, **k: ["se"], raising=False
    )
    assert text.split_into_chunks("x", method="semantic") == ["se"]
    monkeypatch.setattr(
        text, "contextual_chunk_split", lambda *a, **k: ["cx"], raising=False
    )
    assert text.split_into_chunks("x", method="contextual") == ["cx"]
    monkeypatch.setattr(
        text, "summarized_chunk_split", lambda *a, **k: ["su"], raising=False
    )
    assert text.split_into_chunks("x", method="summary") == ["su"]


class DummyQty:
    def __init__(self, value, name):
        self.value = value
        self.unit = types.SimpleNamespace(name=name)
        # include space between number and unit
        self.span = (0, len(str(value)) + len(name) + 1)


class DummyUReg:
    class DummyQuantity:
        def __init__(self, value):
            self.val = value

        def __rmul__(self, other):
            return type(self)(other * self.val)

        def to_base_units(self):
            return types.SimpleNamespace(magnitude=self.val, units="m")

    def __call__(self, name):
        return self.DummyQuantity(1)


def test_normalize_units(monkeypatch):
    q = DummyQty(2, "m")
    monkeypatch.setattr(text, "_PINT_AVAILABLE", True, raising=False)
    monkeypatch.setattr(
        text, "_qty_parser", types.SimpleNamespace(parse=lambda t: [q]), raising=False
    )
    monkeypatch.setattr(text, "_UnitRegistry", DummyUReg, raising=False)
    assert text.normalize_units("2 m") == "2 m"


def test_extract_json_variants():
    assert text.extract_json_from_text('{"a":1}') == {"a": 1}
    md = '```json\n{"b":2}\n```'
    assert text.extract_json_from_text(md) == {"b": 2}
    with pytest.raises(ValueError):
        text.extract_json_from_text("no json")


def test_clean_text(monkeypatch):
    monkeypatch.setattr(text, "_UNSTRUCTURED", False, raising=False)
    monkeypatch.setattr(text, "normalize_units", lambda t: f"<{t}>", raising=False)
    assert text.clean_text(" a  b ") == "<a b>"
    monkeypatch.setattr(text, "_UNSTRUCTURED", True, raising=False)
    monkeypatch.setattr(text, "_clean", lambda t, **k: "X", raising=False)
    assert text.clean_text("anything") == "<X>"


def test_detect_language_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(text, "fasttext", object(), raising=False)
    fake = types.SimpleNamespace(predict=lambda s: (["__label__en"], [0.9]))
    monkeypatch.setattr(text, "get_fasttext", lambda: fake)
    path = tmp_path / "m.bin"
    path.write_text("x")
    res = text.detect_language("hi", model_path=str(path), return_prob=True)
    assert res == ("en", 0.9)

    monkeypatch.setattr(text, "fasttext", None, raising=False)
    assert text.detect_language("hi") == "und"
    assert text.detect_language("hi", return_prob=True) == ("und", 0.0)


def test_get_ft_model_and_release(monkeypatch, tmp_path):
    calls = []

    def fake_load(path):
        calls.append(path)
        return object()

    monkeypatch.setattr(text, "fasttext", types.SimpleNamespace(load_model=fake_load))
    model_path = str(tmp_path / "m.bin")
    _ = text._get_ft_model(model_path)
    again = text._get_ft_model(model_path)
    assert calls == [model_path]
    text._release_ft_model(_)
    text._release_ft_model(again)


def test_get_fasttext_singleton(monkeypatch):
    obj = object()
    calls = []
    monkeypatch.setattr(
        text,
        "fasttext",
        types.SimpleNamespace(load_model=lambda p: (calls.append(p) or obj)),
    )
    first = text.get_fasttext()
    second = text.get_fasttext()
    assert first is second is obj
    assert len(calls) == 1


def test_detect_language_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(text, "fasttext", object(), raising=False)
    monkeypatch.setattr(
        text, "get_fasttext", lambda: types.SimpleNamespace(predict=lambda s: ([], []))
    )
    path = tmp_path / "not.bin"
    with pytest.raises(FileNotFoundError):
        text.detect_language("x", model_path=str(path))
    path.write_text("ok")
    assert text.detect_language("x", model_path=str(path)) == "und"
