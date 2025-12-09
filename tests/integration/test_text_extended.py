import importlib
import types

import pytest

import datacreek.utils.text as text


def test_split_into_chunks_modes(monkeypatch):
    monkeypatch.setattr(text, "sliding_window_chunks", lambda t, c, o: ["sl"])
    assert text.split_into_chunks("x", method="sliding") == ["sl"]

    monkeypatch.setattr(text, "semantic_chunk_split", lambda *a, **k: ["se"])
    assert text.split_into_chunks("x", method="semantic") == ["se"]

    monkeypatch.setattr(text, "contextual_chunk_split", lambda *a, **k: ["ct"])
    assert text.split_into_chunks("x", method="contextual") == ["ct"]

    monkeypatch.setattr(text, "summarized_chunk_split", lambda *a, **k: ["su"])
    assert text.split_into_chunks("x", method="summary") == ["su"]


def test_clean_text_unstructured(monkeypatch):
    monkeypatch.setattr(text, "_UNSTRUCTURED", True, raising=False)
    monkeypatch.setattr(text, "_clean", lambda t, **k: "clean:" + t, raising=False)
    monkeypatch.setattr(text, "normalize_units", lambda s: s + "!")
    assert text.clean_text("hi") == "clean:hi!"


def test_get_release_fasttext(monkeypatch):
    calls = {"n": 0}

    class DummyModel:
        pass

    def load_model(path):
        calls["n"] += 1
        return DummyModel()

    monkeypatch.setattr(text, "fasttext", types.SimpleNamespace(load_model=load_model))
    monkeypatch.setattr(text, "_FASTTEXT_POOL", None, raising=False)
    monkeypatch.setattr(text, "_FT_MODEL", None, raising=False)

    model1 = text._get_ft_model("bin")
    text._release_ft_model(model1)
    model2 = text._get_ft_model("bin")
    text._release_ft_model(model2)

    assert model1 is model2
    assert calls["n"] == 1


def test_detect_language_fallback(monkeypatch):
    monkeypatch.setattr(text, "fasttext", None, raising=False)
    assert text.detect_language("hello") == "und"


def test_detect_language_missing_file(monkeypatch):
    class DummyModel:
        def predict(self, t):
            return ["__label__en"], [1.0]

    monkeypatch.setattr(
        text, "fasttext", types.SimpleNamespace(load_model=lambda p: DummyModel())
    )
    monkeypatch.setattr(text.os.path, "exists", lambda p: False)
    with pytest.raises(FileNotFoundError):
        text.detect_language("hi", model_path="missing")
