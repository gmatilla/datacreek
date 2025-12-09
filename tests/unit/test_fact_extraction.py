import json
import types

import datacreek.utils.fact_extraction as fe


def test_extract_facts_regex():
    text = "Alice is a doctor. Bob was born in Paris."
    facts = fe.extract_facts(text)
    assert {"subject": "Alice", "predicate": "is", "object": "a doctor"} in facts
    assert {"subject": "Bob", "predicate": "born_in", "object": "Paris"} in facts


def test_extract_facts_llm(monkeypatch):
    class DummyClient:
        def chat_completion(self, messages, temperature=0):
            return json.dumps([{"subject": "A", "predicate": "p", "object": "B"}])

    monkeypatch.setattr(fe, "get_prompt", lambda c, n: "prompt")
    monkeypatch.setattr(fe, "load_config", lambda: {})
    client = DummyClient()
    facts = fe.extract_facts(
        "t", client=client, config={"prompts": {"fact_extraction": "x"}}
    )
    assert facts == [{"subject": "A", "predicate": "p", "object": "B"}]


def test_extract_facts_invalid_json(monkeypatch):
    class DummyClient:
        def chat_completion(self, messages, temperature=0):
            return "invalid"

    monkeypatch.setattr(fe, "get_prompt", lambda c, n: "prompt")
    monkeypatch.setattr(fe, "load_config", lambda: {})
    facts = fe.extract_facts(
        "t", client=DummyClient(), config={"prompts": {"fact_extraction": "x"}}
    )
    assert facts == []
