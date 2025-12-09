import importlib

import pytest


def _setup_module(monkeypatch, score):
    """Reload module and patch pipeline/counter."""
    module = importlib.reload(importlib.import_module("ingest.safety_filter"))

    class DummyPipe:
        def __call__(self, text):
            return [{"label": "toxic", "score": score}]

    class DummyCounter:
        def __init__(self):
            self.n = 0

        def inc(self):
            self.n += 1

    monkeypatch.setattr(module, "pipeline", lambda *a, **k: DummyPipe())
    monkeypatch.setattr(module, "_TOXICITY_PIPE", None)
    counter = DummyCounter()
    monkeypatch.setattr(module, "INGEST_TOXIC_BLOCKS", counter)
    return module, counter


def test_blocks_when_regex_and_score_high(monkeypatch):
    module, counter = _setup_module(monkeypatch, 0.9)
    assert module.filter_text("this is porn") is None
    assert counter.n == 1


def test_allows_safe_text(monkeypatch):
    module, counter = _setup_module(monkeypatch, 0.1)
    assert module.filter_text("hello world") == "hello world"
    assert counter.n == 0


def test_regex_low_score_not_blocked(monkeypatch):
    module, counter = _setup_module(monkeypatch, 0.1)
    assert module.filter_text("sex education content") == "sex education content"
    assert counter.n == 0
