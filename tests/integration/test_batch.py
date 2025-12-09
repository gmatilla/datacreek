import asyncio

import pytest

from datacreek.utils.batch import async_process_batches, process_batches


class DummyClient:
    def __init__(self):
        self.calls = []

    async def async_batch_completion(
        self, batches, *, temperature=None, batch_size=None
    ):
        self.calls.append((batches, temperature, batch_size))
        return ["resp:" + batch[0]["content"] for batch in batches]


def test_async_process_batches():
    client = DummyClient()
    messages = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]

    async def run():
        res = await async_process_batches(
            client,
            messages,
            batch_size=1,
            temperature=0.2,
            parse_fn=lambda s: s.upper(),
        )
        assert res == ["RESP:A", "RESP:B"]

    asyncio.run(run())
    assert len(client.calls) == 2


def test_curate_async_mode(monkeypatch):
    async_called = {}

    class DummyClient2:
        def __init__(self):
            self.config = {
                "prompts": {"qa_rating": "{pairs}"},
                "curate": {"batch_size": 1, "temperature": 0.1, "threshold": 0.0},
            }

        async def async_batch_completion(
            self, batches, *, temperature=None, batch_size=None
        ):
            async_called["count"] = len(batches)
            return ['{"rating": 9}'] * len(batches)

    monkeypatch.setattr(
        "datacreek.core.curate.LLMClient", lambda *a, **k: DummyClient2()
    )

    async def fake_async(client, msgs, *, batch_size, temperature, parse_fn, **kwargs):
        async_called["count"] = len(msgs)
        return [parse_fn('{"rating": 9}') for _ in msgs]

    monkeypatch.setattr("datacreek.utils.batch.async_process_batches", fake_async)
    from datacreek.models.qa import QAPair

    monkeypatch.setattr(
        "datacreek.core.curate.parse_ratings",
        lambda resp, orig: [QAPair(question="q", answer="a", rating=9)],
    )
    monkeypatch.setattr(
        "datacreek.core.curate.convert_to_conversation_format", lambda pairs: []
    )

    from datacreek.core.curate import curate_qa_pairs

    data = {"summary": "", "qa_pairs": [{"question": "q", "answer": "a"}]}
    result = curate_qa_pairs(data, async_mode=True)

    assert async_called.get("count") == 1
    assert result["qa_pairs"] == [{"question": "q", "answer": "a", "rating": 9}]


def test_curate_threshold_validation():
    from datacreek.core.curate import curate_qa_pairs

    with pytest.raises(ValueError):
        curate_qa_pairs(
            {"summary": "", "qa_pairs": [{"question": "q", "answer": "a"}]},
            threshold=11,
        )


def test_process_batches_error(monkeypatch):
    class ErrClient:
        def batch_completion(self, *a, **k):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        process_batches(
            ErrClient(),
            [[{"role": "user", "content": "a"}]],
            batch_size=1,
            temperature=0.0,
            parse_fn=lambda s: s,
            raise_on_error=True,
        )


def test_async_process_batches_error(monkeypatch):
    class ErrClient:
        async def async_batch_completion(self, *a, **k):
            raise RuntimeError("boom")

    async def run():
        with pytest.raises(RuntimeError):
            await async_process_batches(
                ErrClient(),
                [[{"role": "user", "content": "a"}]],
                batch_size=1,
                temperature=0.0,
                parse_fn=lambda s: s,
                raise_on_error=True,
            )

    asyncio.run(run())
