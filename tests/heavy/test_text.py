import importlib
import os
import types

import pytest

import datacreek.utils.text as text


def reload_text_module():
    return importlib.reload(text)


class DummyModel:
    def __init__(self):
        self.called = []

    def predict(self, txt):
        self.called.append(txt)
        return ["__label__en"], [0.75]


def test_split_into_chunks_methods(monkeypatch):
    called = {}

    def stub(name):
        def fn(*args, **kwargs):
            called[name] = True
            return [name]

        return fn

    monkeypatch.setattr(text, "sliding_window_chunks", stub("sliding"))
    monkeypatch.setattr(text, "semantic_chunk_split", stub("semantic"))
    monkeypatch.setattr(text, "contextual_chunk_split", stub("contextual"))
    monkeypatch.setattr(text, "summarized_chunk_split", stub("summary"))
    assert text.split_into_chunks("a b c", method="sliding") == ["sliding"]
    assert called["sliding"]
    assert text.split_into_chunks("a", method="semantic") == ["semantic"]
    assert text.split_into_chunks("b", method="contextual") == ["contextual"]
    assert text.split_into_chunks("c", method="summary") == ["summary"]
    # default logic
    parts = text.split_into_chunks("p1\n\np2\n\np3", chunk_size=4)
    assert len(parts) == 2
    assert parts[-1] == "p3"


def test_normalize_units(monkeypatch):
    module = reload_text_module()
    monkeypatch.setattr(module, "_PINT_AVAILABLE", False)
    assert module.normalize_units("5 km") == "5 km"

    class DummyUnit:
        def __init__(self, name):
            self.name = name

        def __rmul__(self, value):
            return DummyQty(value, self.name)

    class DummyQty:
        def __init__(self, magnitude, units):
            self.magnitude = magnitude
            self.units = units

        def to_base_units(self):
            if self.units == "km":
                return DummyQty(self.magnitude * 1000, "meter")
            return self

    class DummyRegistry:
        def __call__(self, name):
            return DummyUnit(name)

    class DummyParser:
        def parse(self, text):
            return [
                types.SimpleNamespace(
                    value=5, unit=types.SimpleNamespace(name="km"), span=(0, 4)
                )
            ]

    module = reload_text_module()
    monkeypatch.setattr(module, "_PINT_AVAILABLE", True)
    monkeypatch.setattr(module, "_UnitRegistry", DummyRegistry, raising=False)
    monkeypatch.setattr(module, "_qty_parser", DummyParser(), raising=False)
    assert module.normalize_units("5 km run") == "5000 meter run"


def test_extract_json_from_text():
    assert text.extract_json_from_text('{"a": 1}') == {"a": 1}
    md = '```json\n{"a":2}\n```'
    assert text.extract_json_from_text(md) == {"a": 2}
    partial = 'prefix {"b":3} suffix'
    assert text.extract_json_from_text(partial) == {"b": 3}
    with pytest.raises(ValueError):
        text.extract_json_from_text("no json here")


def test_clean_text(monkeypatch):
    module = reload_text_module()
    monkeypatch.setattr(module, "_UNSTRUCTURED", True)
    monkeypatch.setattr(module, "_clean", lambda s, **kw: "A   B", raising=False)
    monkeypatch.setattr(module, "normalize_units", lambda s: s.lower())
    assert module.clean_text("X") == "a   b".lower()
    module = reload_text_module()
    monkeypatch.setattr(module, "_UNSTRUCTURED", False)
    assert module.clean_text("A\nB") == "A B"


def test_fasttext_model_pool(monkeypatch):
    module = reload_text_module()
    dm = DummyModel()
    monkeypatch.setattr(
        module, "fasttext", types.SimpleNamespace(load_model=lambda p: dm)
    )
    monkeypatch.setattr(module.os, "cpu_count", lambda: 1)
    module._FASTTEXT_POOL = None
    module._FT_MODEL = None
    m = module._get_ft_model("x")
    assert m is dm
    assert module._FASTTEXT_POOL.qsize() == 0
    module._release_ft_model(m)
    assert module._FASTTEXT_POOL.qsize() == 1


def test_get_fasttext_cached(monkeypatch):
    module = reload_text_module()
    dm = DummyModel()
    called = {}

    def loader(path):
        called[path] = called.get(path, 0) + 1
        return dm

    monkeypatch.setattr(module, "fasttext", types.SimpleNamespace(load_model=loader))
    if hasattr(module.get_fasttext, "_model"):
        delattr(module.get_fasttext, "_model")
    m1 = module.get_fasttext()
    m2 = module.get_fasttext()
    assert m1 is dm and m2 is dm
    assert called[module.FASTTEXT_BIN] == 1


def test_detect_language(monkeypatch):
    module = reload_text_module()
    dm = DummyModel()
    dm.predict = lambda text: (["__label__fr"], [0.9])
    monkeypatch.setattr(
        module, "fasttext", types.SimpleNamespace(load_model=lambda p: dm)
    )
    monkeypatch.setattr(module.os.path, "exists", lambda p: True)
    monkeypatch.setattr(module, "get_fasttext", lambda: dm)
    assert module.detect_language("bonjour", return_prob=True) == ("fr", 0.9)
    dm.predict = lambda text: ([], [])
    assert module.detect_language("", return_prob=False) == "und"
    monkeypatch.setattr(module, "fasttext", None)
    assert module.detect_language("hi") == "und"
    module = reload_text_module()
    dm2 = DummyModel()
    monkeypatch.setattr(
        module, "fasttext", types.SimpleNamespace(load_model=lambda p: dm2)
    )
    monkeypatch.setattr(module.os.path, "exists", lambda p: False)
    monkeypatch.setattr(module, "get_fasttext", lambda: dm2)
    with pytest.raises(FileNotFoundError):
        module.detect_language("text")
