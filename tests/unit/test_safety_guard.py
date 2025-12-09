import importlib
import sys
from pathlib import Path

# Ensure project root on sys.path for direct module imports
sys.path.append(str(Path(__file__).resolve().parents[2]))


def _setup(monkeypatch, tox_score=0.0, nsfw_score=0.0):
    module = importlib.reload(importlib.import_module("ingest.safety_guard"))

    class DummyCounter:
        def __init__(self):
            self.n = 0

        def inc(self):
            self.n += 1

    monkeypatch.setattr(module, "_toxicity_score", lambda text: tox_score)
    monkeypatch.setattr(module, "_nsfw_image_score", lambda img: nsfw_score)
    counter = DummyCounter()
    monkeypatch.setattr(module, "INGEST_TOXIC_BLOCKS", counter)
    return module, counter


def test_blocks_on_regex(monkeypatch):
    module, counter = _setup(monkeypatch)
    assert module.guard("porn content") is None
    assert counter.n == 1


def test_blocks_when_score_high(monkeypatch):
    module, counter = _setup(monkeypatch, tox_score=0.8, nsfw_score=0.8)
    assert module.guard("harmless") is None
    assert counter.n == 1


def test_allows_safe_payload(monkeypatch):
    module, counter = _setup(monkeypatch, tox_score=0.1, nsfw_score=0.0)
    assert module.guard("hello world") == "hello world"
    assert counter.n == 0
