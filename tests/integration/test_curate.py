import json

import pytest

from datacreek.core import curate
from datacreek.models.qa import QAPair


class DummyClient:
    def __init__(self, *a, **k):
        self.config = {}


def fake_batch(client, message_batches, **kwargs):
    res = []
    for m in message_batches:
        pairs = json.loads(m[0]["content"])
        res.append(
            json.dumps(
                [
                    {"question": p["question"], "answer": p["answer"], "rating": 10}
                    for p in pairs
                ]
            )
        )
    return res


def fake_parse(text, orig):
    data = json.loads(text)
    return [
        QAPair(question=d["question"], answer=d["answer"], rating=d["rating"])
        for d in data
    ]


class DummyProgress:
    def __init__(self, *a, **k):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def update(self, *a, **k):
        pass


def test_curate_deduplicate(monkeypatch):
    monkeypatch.setattr(curate, "LLMClient", DummyClient)
    monkeypatch.setattr("datacreek.utils.batch.process_batches", fake_batch)
    monkeypatch.setattr(curate, "parse_ratings", fake_parse)
    monkeypatch.setattr(
        "datacreek.utils.progress.create_progress", lambda *a, **k: (DummyProgress(), 0)
    )
    monkeypatch.setattr(curate, "get_prompt", lambda cfg, name: "{pairs}")
    data = {
        "summary": "",
        "qa_pairs": [
            {"question": "q", "answer": "a"},
            {"question": "q", "answer": "a"},
        ],
    }
    res = curate.curate_qa_pairs(data, batch_size=1, inference_batch=1)
    assert len(res["qa_pairs"]) == 1


def test_curate_parse_error(monkeypatch):
    """Parsing failures should raise CurationError."""

    monkeypatch.setattr(curate, "LLMClient", DummyClient)
    monkeypatch.setattr("datacreek.utils.batch.process_batches", fake_batch)

    def bad_parse(text, orig):
        raise ValueError("nope")

    monkeypatch.setattr(curate, "parse_ratings", bad_parse)
    monkeypatch.setattr(
        "datacreek.utils.progress.create_progress", lambda *a, **k: (DummyProgress(), 0)
    )
    monkeypatch.setattr(curate, "get_prompt", lambda cfg, name: "{pairs}")

    data = {"summary": "", "qa_pairs": [{"question": "q", "answer": "a"}]}

    with pytest.raises(curate.CurationError):
        curate.curate_qa_pairs(data, batch_size=1, inference_batch=1)


def test_curate_resume(monkeypatch, tmp_path):
    monkeypatch.setattr(curate, "LLMClient", DummyClient)
    monkeypatch.setattr("datacreek.utils.batch.process_batches", fake_batch)
    monkeypatch.setattr(curate, "parse_ratings", fake_parse)
    monkeypatch.setattr(
        "datacreek.utils.progress.create_progress", lambda *a, **k: (DummyProgress(), 0)
    )
    monkeypatch.setattr(curate, "get_prompt", lambda cfg, name: "{pairs}")

    existing = {
        "summary": "",
        "qa_pairs": [],
        "conversations": [],
        "metrics": {"total": 1, "filtered": 1, "retention_rate": 1.0, "avg_score": 10},
        "rated_pairs": [{"question": "q1", "answer": "a1", "rating": 10}],
    }
    data = {
        "summary": "",
        "qa_pairs": [
            {"question": "q1", "answer": "a1"},
            {"question": "q2", "answer": "a2"},
        ],
    }

    res = curate.curate_qa_pairs(
        data,
        batch_size=1,
        inference_batch=1,
        resume=True,
        previous_result=existing,
    )
    assert len(res["qa_pairs"]) == 2


def test_curate_as_dataclass(monkeypatch):
    monkeypatch.setattr(curate, "LLMClient", DummyClient)
    monkeypatch.setattr("datacreek.utils.batch.process_batches", fake_batch)
    monkeypatch.setattr(curate, "parse_ratings", fake_parse)
    monkeypatch.setattr(
        "datacreek.utils.progress.create_progress", lambda *a, **k: (DummyProgress(), 0)
    )
    monkeypatch.setattr(curate, "get_prompt", lambda cfg, name: "{pairs}")

    data = {"summary": "", "qa_pairs": [{"question": "q", "answer": "a"}]}

    res = curate.curate_qa_pairs(
        data, batch_size=1, inference_batch=1, as_dataclass=True
    )
    assert isinstance(res, curate.CurationResult)


def test_filter_rated_pairs():
    pairs = [
        QAPair(question="q1", answer="a1", rating=9),
        QAPair(question="q2", answer="a2", rating=5),
    ]
    filtered = curate.filter_rated_pairs(pairs, threshold=7)
    assert len(filtered) == 1
    assert filtered[0].question == "q1"


def test_apply_curation_threshold():
    result = {
        "summary": "",
        "qa_pairs": [],
        "conversations": [],
        "metrics": {"total": 2, "filtered": 1, "retention_rate": 0.5, "avg_score": 7.5},
        "rated_pairs": [
            {"question": "q1", "answer": "a1", "rating": 8},
            {"question": "q2", "answer": "a2", "rating": 6},
        ],
    }
    new_res = curate.apply_curation_threshold(result, 7)
    assert len(new_res.qa_pairs) == 1
    assert new_res.qa_pairs[0].question == "q1"
    assert new_res.metrics.filtered == 1
