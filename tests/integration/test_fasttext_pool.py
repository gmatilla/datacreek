import importlib.util
from pathlib import Path

root = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "datacreek.utils.text", root / "datacreek" / "utils" / "text.py"
)
text = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(text)


def test_fasttext_pool_single_load(monkeypatch):
    calls = {"n": 0}

    class DummyModel:
        def predict(self, txt):
            return ["__label__en"], [0.9]

    class DummyFastText:
        def load_model(self, path):
            calls["n"] += 1
            return DummyModel()

    monkeypatch.setattr(text, "fasttext", DummyFastText())
    monkeypatch.setattr(text.os.path, "exists", lambda p: True)
    if hasattr(text.get_fasttext, "_model"):
        delattr(text.get_fasttext, "_model")

    lang1, p1 = text.detect_language("hello", model_path="lid.bin", return_prob=True)
    lang2, p2 = text.detect_language("world", model_path="lid.bin", return_prob=True)

    assert lang1 == "en" and p1 == 0.9
    assert calls["n"] == 1
    assert lang2 == "en" and p2 == 0.9
    assert calls["n"] == 1
