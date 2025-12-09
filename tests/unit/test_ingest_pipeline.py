import importlib
import sys
from pathlib import Path

# Ensure project root on sys.path for direct module imports
sys.path.append(str(Path(__file__).resolve().parents[2]))


class DummyCounter:
    def __init__(self):
        self.n = 0

    def inc(self):
        self.n += 1


def test_language_skip(monkeypatch):
    lang_mod = importlib.reload(importlib.import_module("datacreek.utils.lang_detect"))
    sg_mod = importlib.reload(importlib.import_module("ingest.safety_guard"))
    monkeypatch.setattr(lang_mod, "LANG_SKIPPED_TOTAL", DummyCounter())
    monkeypatch.setattr(lang_mod, "detect_language", lambda text: "es")
    monkeypatch.setattr(sg_mod, "INGEST_TOXIC_BLOCKS", DummyCounter())
    pipeline = importlib.reload(importlib.import_module("ingest.pipeline"))
    assert pipeline.safe_ingest("hola", {"en", "fr"}) is None
    assert lang_mod.LANG_SKIPPED_TOTAL.n == 1
    assert sg_mod.INGEST_TOXIC_BLOCKS.n == 0


def test_toxic_block(monkeypatch):
    lang_mod = importlib.reload(importlib.import_module("datacreek.utils.lang_detect"))
    sg_mod = importlib.reload(importlib.import_module("ingest.safety_guard"))
    monkeypatch.setattr(lang_mod, "LANG_SKIPPED_TOTAL", DummyCounter())
    monkeypatch.setattr(lang_mod, "detect_language", lambda text: "en")
    monkeypatch.setattr(sg_mod, "INGEST_TOXIC_BLOCKS", DummyCounter())
    monkeypatch.setattr(sg_mod, "_toxicity_score", lambda text: 0.9)
    monkeypatch.setattr(sg_mod, "_nsfw_image_score", lambda img: 0.9)
    pipeline = importlib.reload(importlib.import_module("ingest.pipeline"))
    assert pipeline.safe_ingest("bad", {"en", "fr"}) is None
    assert sg_mod.INGEST_TOXIC_BLOCKS.n == 1


def test_safe_pass(monkeypatch):
    lang_mod = importlib.reload(importlib.import_module("datacreek.utils.lang_detect"))
    sg_mod = importlib.reload(importlib.import_module("ingest.safety_guard"))
    monkeypatch.setattr(lang_mod, "LANG_SKIPPED_TOTAL", DummyCounter())
    monkeypatch.setattr(lang_mod, "detect_language", lambda text: "fr")
    monkeypatch.setattr(sg_mod, "INGEST_TOXIC_BLOCKS", DummyCounter())
    monkeypatch.setattr(sg_mod, "_toxicity_score", lambda text: 0.1)
    monkeypatch.setattr(sg_mod, "_nsfw_image_score", lambda img: 0.0)
    pipeline = importlib.reload(importlib.import_module("ingest.pipeline"))
    assert pipeline.safe_ingest("bonjour", {"en", "fr"}) == "bonjour"
    assert sg_mod.INGEST_TOXIC_BLOCKS.n == 0
