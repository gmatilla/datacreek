import types

import pytest

import datacreek.utils.text as text


def test_normalize_units_conversion(monkeypatch):
    class DummyParser:
        def parse(self, t):
            return [
                types.SimpleNamespace(
                    value=1, unit=types.SimpleNamespace(name="km"), span=(0, 2)
                )
            ]

    class DummyQty:
        def __init__(self, mag):
            self.mag = mag

        def to_base_units(self):
            return types.SimpleNamespace(magnitude=self.mag * 1000, units="meter")

        def __rmul__(self, other):
            return DummyQty(other)

    class DummyRegistry:
        def __call__(self, unit):
            return DummyQty(1)

    monkeypatch.setattr(text, "_PINT_AVAILABLE", True, raising=False)
    monkeypatch.setattr(text, "_UnitRegistry", DummyRegistry, raising=False)
    monkeypatch.setattr(text, "_qty_parser", DummyParser(), raising=False)
    out = text.normalize_units("1 km")
    assert out.startswith("1000") and "meter" in out


def test_clean_text_fallback(monkeypatch):
    monkeypatch.setattr(text, "_UNSTRUCTURED", False, raising=False)
    assert text.clean_text("A  B") == "A B"


def test_detect_language_success(monkeypatch):
    class DummyModel:
        def predict(self, t):
            return ["__label__fr"], [0.9]

    monkeypatch.setattr(
        text, "fasttext", types.SimpleNamespace(load_model=lambda p: DummyModel())
    )
    monkeypatch.setattr(text.os.path, "exists", lambda p: True)
    monkeypatch.setattr(text, "get_fasttext", lambda: DummyModel())
    lang, prob = text.detect_language("bonjour", return_prob=True)
    assert lang == "fr" and prob == 0.9


def test_normalize_units_no_pint(monkeypatch):
    """Fall back to original text when pint is unavailable."""
    monkeypatch.setattr(text, "_PINT_AVAILABLE", False, raising=False)
    assert text.normalize_units("abc") == "abc"


def test_extract_json_from_text(monkeypatch):
    """Extract JSON from various text formats."""
    assert text.extract_json_from_text('{"a": 1}') == {"a": 1}

    md = 'pre```json\n{"b":2}\n```post'
    assert text.extract_json_from_text(md) == {"b": 2}

    assert text.extract_json_from_text('prefix {"c":3} suffix') == {"c": 3}

    with pytest.raises(ValueError):
        text.extract_json_from_text("no json here")


def test_split_into_chunks_default():
    """Default chunk splitting joins small paragraphs."""
    text_in = "a\n\nb\n\nc"
    assert text.split_into_chunks(text_in, chunk_size=2) == ["a\n\nb", "c"]
