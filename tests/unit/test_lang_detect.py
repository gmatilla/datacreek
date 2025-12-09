import importlib
import sys
from pathlib import Path

# Ensure project root on sys.path for direct module imports
sys.path.append(str(Path(__file__).resolve().parents[2]))


def _setup(monkeypatch):
    module = importlib.reload(importlib.import_module("datacreek.utils.lang_detect"))

    class DummyCounter:
        def __init__(self):
            self.n = 0

        def inc(self):
            self.n += 1

    counter = DummyCounter()
    monkeypatch.setattr(module, "LANG_SKIPPED_TOTAL", counter)
    return module, counter


def test_skips_disallowed_language(monkeypatch):
    module, counter = _setup(monkeypatch)
    assert module.should_process("hola que tal", {"en", "fr"}) is False
    assert counter.n == 1


def test_allows_supported_language(monkeypatch):
    module, counter = _setup(monkeypatch)
    assert module.should_process("bonjour tout le monde", {"en", "fr"}) is True
    assert counter.n == 0
