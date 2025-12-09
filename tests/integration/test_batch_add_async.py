import asyncio

from datacreek.generators.qa_generator import QAGenerator


class DummyClient:
    def __init__(self):
        self.config = {
            "prompts": {"qa_rating": "{pairs}", "qa_generation": "prompt"},
            "curate": {"batch_size": 1, "temperature": 0.1, "threshold": 0.0},
        }


async_called = {}


async def fake_async(client, messages, *, batch_size, temperature, parse_fn, **kwargs):
    async_called["count"] = len(messages)
    return ["{}"] * len(messages)


def test_rate_qa_pairs_async(monkeypatch):
    monkeypatch.setattr("datacreek.utils.batch.async_process_batches", fake_async)
    from datacreek.models.qa import QAPair

    monkeypatch.setattr(
        "datacreek.generators.qa_generator.parse_ratings",
        lambda resp, orig: [QAPair(question="q", answer="a", rating=9)],
    )

    gen = QAGenerator(DummyClient())
    pairs, metrics = gen.rate_qa_pairs(
        [{"question": "q", "answer": "a"}], "", async_mode=True
    )

    assert async_called.get("count") == 1
    assert pairs == [{"question": "q", "answer": "a", "rating": 9}]


def test_generate_qa_pairs_async(monkeypatch):
    async_called.clear()

    async def fake_async2(
        client, messages, *, batch_size, temperature, parse_fn, **kwargs
    ):
        async_called["count"] = len(messages)
        return [parse_fn('[{"question": "q", "answer": "a"}]') for _ in messages]

    monkeypatch.setattr("datacreek.utils.batch.async_process_batches", fake_async2)
    from datacreek.models.qa import QAPair

    monkeypatch.setattr(
        "datacreek.generators.qa_generator.parse_qa_pairs",
        lambda resp: [QAPair(question="q", answer="a")],
    )

    gen = QAGenerator(DummyClient())
    pairs = gen.generate_qa_pairs("doc", "sum", num_pairs=1, async_mode=True)

    assert async_called.get("count") == 1
    assert pairs == [QAPair(question="q", answer="a", chunk="doc", source="inline")]


def test_generate_qa_pairs_async_direct(monkeypatch):
    async_called.clear()

    async def fake_async3(
        client, messages, *, batch_size, temperature, parse_fn, **kwargs
    ):
        async_called["count"] = len(messages)
        return [parse_fn('[{"question": "q", "answer": "a"}]') for _ in messages]

    monkeypatch.setattr("datacreek.utils.batch.async_process_batches", fake_async3)
    from datacreek.models.qa import QAPair

    monkeypatch.setattr(
        "datacreek.generators.qa_generator.parse_qa_pairs",
        lambda resp: [QAPair(question="q", answer="a")],
    )

    gen = QAGenerator(DummyClient())
    pairs = asyncio.run(gen.generate_qa_pairs_async("doc", "sum", num_pairs=1))

    assert async_called.get("count") == 1
    assert pairs == [QAPair(question="q", answer="a", chunk="doc", source="inline")]
