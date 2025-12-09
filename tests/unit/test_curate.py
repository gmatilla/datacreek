import asyncio
import json
import types

import pytest

from datacreek.core import curate
from datacreek.models.qa import QAPair
from datacreek.models.results import CurationMetrics, CurationResult


class DummyClient:
    def __init__(self):
        self.config = {
            "curate": {
                "threshold": 7.0,
                "batch_size": 2,
                "inference_batch": 1,
                "temperature": 0.1,
            },
            "prompts": {"qa_rating": "rate {pairs}"},
        }

    async def async_batch_completion(self, msgs, *, temperature, batch_size):
        # always return one rated pair
        return [json.dumps({"question": "Q1", "answer": "A1", "rating": 8})]

    def batch_completion(self, msgs, *, temperature, batch_size):
        return asyncio.run(
            self.async_batch_completion(
                msgs, temperature=temperature, batch_size=batch_size
            )
        )


def setup_dummy(monkeypatch):
    client = DummyClient()
    monkeypatch.setattr(curate, "LLMClient", lambda **_: client)
    monkeypatch.setattr(
        curate,
        "get_curate_settings",
        lambda cfg: types.SimpleNamespace(**cfg["curate"]),
    )
    monkeypatch.setattr(curate, "get_prompt", lambda cfg, n: cfg["prompts"][n])

    async def fake_async(*a, **k):
        return await client.async_batch_completion([], temperature=0, batch_size=1)

    def fake_sync(*a, **k):
        return [json.dumps({"question": "Q1", "answer": "A1", "rating": 8})]

    monkeypatch.setattr("datacreek.utils.batch.async_process_batches", fake_async)
    monkeypatch.setattr("datacreek.utils.batch.process_batches", fake_sync)


@pytest.mark.asyncio
async def test_curate_qa_pairs_async(monkeypatch):
    setup_dummy(monkeypatch)
    data = {"qa_pairs": [{"question": "Q1", "answer": "A1"}], "summary": "s"}
    res = await curate.curate_qa_pairs_async(data, threshold=7, verbose=False)
    assert isinstance(res, dict)
    assert res["metrics"]["filtered"] == 1
    assert res["metrics"]["total"] == 1


@pytest.mark.asyncio
async def test_curate_qa_pairs_async_invalid(monkeypatch):
    setup_dummy(monkeypatch)
    with pytest.raises(ValueError):
        await curate.curate_qa_pairs_async({"qa_pairs": []})
    with pytest.raises(ValueError):
        await curate.curate_qa_pairs_async(
            {"qa_pairs": [{"question": "q", "answer": "a"}]}, threshold=11
        )


@pytest.mark.asyncio
async def test_curate_qa_pairs_async_parse_error(monkeypatch):
    setup_dummy(monkeypatch)

    async def bad_async(*a, **k):
        return ["not json"]

    monkeypatch.setattr("datacreek.utils.batch.async_process_batches", bad_async)
    with pytest.raises(curate.CurationError):
        await curate.curate_qa_pairs_async(
            {"qa_pairs": [{"question": "q", "answer": "a"}]}
        )


@pytest.mark.asyncio
async def test_curate_qa_pairs_async_resume(monkeypatch):
    setup_dummy(monkeypatch)
    prev = json.dumps(
        {"rated_pairs": [{"question": "Q1", "answer": "A1", "rating": 8}]}
    )
    res = await curate.curate_qa_pairs_async(
        {"qa_pairs": [{"question": "Q2", "answer": "A2"}]},
        threshold=7,
        resume=True,
        previous_result=prev,
        keep_ratings=True,
        as_dataclass=True,
    )
    assert len(res.rated_pairs) == 2
    assert res.metrics.total == 1


def test_filter_rated_pairs_and_threshold():
    pairs = [
        QAPair(question="q", answer="a", rating=5),
        QAPair(question="q2", answer="a2", rating=9),
    ]
    filtered = curate.filter_rated_pairs(pairs, 6)
    assert filtered == [pairs[1]]
    with pytest.raises(ValueError):
        curate.filter_rated_pairs(pairs, -1)


def test_apply_curation_threshold(monkeypatch):
    pairs = [
        QAPair(question="q", answer="a", rating=5),
        QAPair(question="q", answer="a", rating=7),
    ]
    res_dict = {
        "summary": "s",
        "metrics": {"total": 2, "filtered": 0, "retention_rate": 0.0, "avg_score": 6},
        "rated_pairs": [p.to_dict() for p in pairs],
    }
    monkeypatch.setattr(curate, "deduplicate_pairs", lambda p: [p[0]])
    new = curate.apply_curation_threshold(res_dict, 6)
    assert isinstance(new, CurationResult)
    assert new.metrics.filtered == 1
    assert len(new.qa_pairs) == 1


def test_apply_curation_threshold_dataclass(monkeypatch):
    pairs = [QAPair(question="q", answer="a", rating=9)]
    res = CurationResult(
        summary="s",
        qa_pairs=[],
        conversations=[],
        metrics=CurationMetrics(total=1, filtered=0, retention_rate=0.0, avg_score=9),
        rated_pairs=pairs,
    )
    monkeypatch.setattr(curate, "deduplicate_pairs", lambda p: p)
    out = curate.apply_curation_threshold(res, 8)
    assert out.metrics.filtered == 1


def test_curate_qa_pairs_sync_uses_impl(monkeypatch):
    called = {}

    async def fake_impl(*args, **kwargs):
        called["ok"] = True
        return {"metrics": {"filtered": 0, "total": 0}}

    monkeypatch.setattr(curate, "_curate_qa_pairs_impl", fake_impl)
    result = curate.curate_qa_pairs({"qa_pairs": [{"question": "q", "answer": "a"}]})
    assert called.get("ok") and result["metrics"]["total"] == 0


@pytest.mark.asyncio
async def test_curate_qa_pairs_async_input_errors(monkeypatch):
    setup_dummy(monkeypatch)
    with pytest.raises(TypeError):
        await curate.curate_qa_pairs_async(123)


@pytest.mark.asyncio
async def test_curate_from_file(monkeypatch, tmp_path):
    setup_dummy(monkeypatch)
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps({"qa_pairs": [{"question": "q", "answer": "a"}]}))
    res = await curate._curate_qa_pairs_impl(
        str(path),
        threshold=7,
        api_base=None,
        model=None,
        config_path=None,
        verbose=False,
        provider=None,
        kg=None,
        batch_size=None,
        inference_batch=None,
        keep_ratings=False,
        temperature=None,
        resume=False,
        previous_result=None,
        as_dataclass=False,
        use_async_handlers=False,
    )
    assert res["metrics"]["filtered"] == 1


@pytest.mark.asyncio
async def test_curate_resume_invalid_json(monkeypatch):
    setup_dummy(monkeypatch)
    data = {"qa_pairs": [{"question": "q", "answer": "a"}]}
    res = await curate._curate_qa_pairs_impl(
        data,
        threshold=7,
        api_base=None,
        model=None,
        config_path=None,
        verbose=False,
        provider=None,
        kg=None,
        batch_size=None,
        inference_batch=None,
        keep_ratings=False,
        temperature=None,
        resume=True,
        previous_result="{invalid",
        as_dataclass=True,
        use_async_handlers=True,
    )
    assert isinstance(res, CurationResult)
    assert res.metrics.total == 1
