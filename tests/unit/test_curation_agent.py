import json
import types

import pytest

from datacreek.utils.curation_agent import (
    fine_tune_from_feedback,
    propose_merge_split,
    record_feedback,
)


class DummyLLM:
    def __init__(self, response):
        self.response = response

    def predict(self, prompt):
        return self.response


def test_propose_merge_split(monkeypatch, tmp_path):
    llm = DummyLLM('{"merge": ["a"], "split": ["b"]}')
    res = propose_merge_split(["a", "b"], llm=llm)
    assert res == {"merge": ["a"], "split": ["b"]}
    path = tmp_path / "fb.jsonl"
    record_feedback(res, True, path)
    data = path.read_text().strip()
    assert json.loads(data)["accepted"] is True


def test_fine_tune(monkeypatch, tmp_path):
    calls = {}

    class DummyOpenAI:
        class FineTuningJob:
            @staticmethod
            def create(training_file, model):
                calls["file"] = training_file
                calls["model"] = model
                return "ok"

    monkeypatch.setattr("datacreek.utils.curation_agent.openai", DummyOpenAI)
    result = fine_tune_from_feedback("f.jsonl", model="gpt")
    assert result == "ok"
    assert calls["file"] == "f.jsonl"
    assert calls["model"] == "gpt"


def test_bad_response():
    llm = DummyLLM("not json")
    with pytest.raises(ValueError):
        propose_merge_split(["a"], llm=llm)
