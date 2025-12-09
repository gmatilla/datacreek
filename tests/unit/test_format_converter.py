import json
import sys
import types

import pytest

import datacreek.utils.format_converter as fc


def test_to_jsonl_basic():
    data = [{"foo": 1}, {"bar": "x"}]
    result = fc.to_jsonl(data)
    assert result.splitlines() == [json.dumps(d, ensure_ascii=False) for d in data]


def test_to_alpaca_and_fine_tuning():
    qa_pairs = [{"question": "Q?", "answer": "A!"}]
    alpaca = json.loads(fc.to_alpaca(qa_pairs))
    assert alpaca == [{"instruction": "Q?", "input": "", "output": "A!"}]

    ft = json.loads(fc.to_fine_tuning(qa_pairs))
    assert ft[0]["messages"][1]["content"] == "Q?"
    assert ft[0]["messages"][2]["content"] == "A!"


def test_to_chatml():
    qa_pairs = [
        {"question": "q1", "answer": "a1"},
        {"question": "q2", "answer": "a2"},
    ]
    out = fc.to_chatml(qa_pairs).splitlines()
    msgs = [json.loads(line)["messages"][1]["content"] for line in out]
    assert msgs == ["q1", "q2"]


def test_to_hf_dataset_success(monkeypatch):
    class DummyDataset:
        def __init__(self, data):
            self.data = data

        @classmethod
        def from_dict(cls, d):
            return cls(d)

        def to_json(self):
            return json.dumps(self.data)

    dummy_module = types.SimpleNamespace(Dataset=DummyDataset)
    monkeypatch.setitem(sys.modules, "datasets", dummy_module)
    qa_pairs = [{"question": "q", "answer": "a"}]
    result = fc.to_hf_dataset(qa_pairs)
    assert json.loads(result) == {"question": ["q"], "answer": ["a"]}


def test_to_hf_dataset_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(ImportError):
        fc.to_hf_dataset([{"question": "q", "answer": "a"}])
