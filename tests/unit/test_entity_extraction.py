import sys
import types

import datacreek.utils.entity_extraction as ee


def test_extract_entities_regex(monkeypatch):
    class DummySpacy:
        def load(self, model):
            raise RuntimeError

    monkeypatch.setitem(sys.modules, "spacy", DummySpacy())
    text = "John Doe went to Paris."
    result = ee.extract_entities(text)
    assert sorted(result) == ["John Doe", "Paris"]


def test_extract_entities_spacy(monkeypatch):
    class DummyDoc:
        def __init__(self, text):
            self.ents = [types.SimpleNamespace(text="Entity")]

    class DummySpacy:
        def load(self, model):
            def nlp(t):
                return DummyDoc(t)

            return nlp

    monkeypatch.setitem(sys.modules, "spacy", DummySpacy())
    assert ee.extract_entities("whatever") == ["Entity"]
