import logging

import pytest

from datacreek.utils import batch


class DummyClient:
    def __init__(self, *, raise_error=False):
        self.raise_error = raise_error
        self.calls = []

    def batch_completion(self, msgs, *, temperature, batch_size):
        self.calls.append(("sync", len(msgs)))
        if self.raise_error:
            raise RuntimeError("boom")
        return [f"ok{len(msgs)}"] * len(msgs)

    async def async_batch_completion(self, msgs, *, temperature, batch_size):
        self.calls.append(("async", len(msgs)))
        if self.raise_error:
            raise RuntimeError("boom")
        return [f"ok{len(msgs)}"] * len(msgs)


def test_process_batches_success(caplog):
    client = DummyClient()
    parse = lambda x: x + "_parsed"
    msgs = [[{"m": "a"}], [{"m": "b"}]]
    with caplog.at_level(logging.ERROR):
        res = batch.process_batches(
            client, msgs, batch_size=1, temperature=0.0, parse_fn=parse
        )
    assert res == ["ok1_parsed", "ok1_parsed"]
    assert client.calls == [("sync", 1), ("sync", 1)]
    assert not caplog.records


def test_process_batches_parse_error(caplog):
    client = DummyClient()

    def bad_parse(x):
        raise ValueError("bad")

    with caplog.at_level(logging.ERROR):
        out = batch.process_batches(
            client, [[{"a": 1}]], batch_size=1, temperature=0.0, parse_fn=bad_parse
        )
    assert out == []
    assert "Failed to parse response" in caplog.text


def test_process_batches_raise_on_error():
    client = DummyClient(raise_error=True)
    with pytest.raises(RuntimeError):
        batch.process_batches(
            client,
            [[{"a": 1}]],
            batch_size=1,
            temperature=0.0,
            parse_fn=str,
            raise_on_error=True,
        )


def test_process_batches_log_error(caplog):
    client = DummyClient(raise_error=True)
    with caplog.at_level(logging.ERROR):
        res = batch.process_batches(
            client, [[{"a": 1}]], batch_size=1, temperature=0.0, parse_fn=str
        )
    assert res == []
    assert "Error processing batch" in caplog.text


@pytest.mark.asyncio
async def test_async_process_batches_success():
    client = DummyClient()
    res = await batch.async_process_batches(
        client,
        [[{"m": 1}], [{"m": 2}]],
        batch_size=1,
        temperature=0.0,
        parse_fn=lambda x: x,
    )
    assert res == ["ok1", "ok1"]
    assert client.calls == [("async", 1), ("async", 1)]


@pytest.mark.asyncio
async def test_async_process_batches_error():
    client = DummyClient(raise_error=True)
    res = await batch.async_process_batches(
        client, [[{"a": 1}]], batch_size=1, temperature=0.0, parse_fn=lambda x: x
    )
    assert res == []


@pytest.mark.asyncio
async def test_async_process_batches_parse_error(caplog):
    client = DummyClient()

    def bad_parse(_):
        raise ValueError("bad")

    with caplog.at_level(logging.ERROR):
        out = await batch.async_process_batches(
            client,
            [[{"a": 1}]],
            batch_size=1,
            temperature=0.0,
            parse_fn=bad_parse,
        )
    assert out == []
    assert "Failed to parse response" in caplog.text


@pytest.mark.asyncio
async def test_async_process_batches_raise_on_error():
    client = DummyClient(raise_error=True)
    with pytest.raises(RuntimeError):
        await batch.async_process_batches(
            client,
            [[{"a": 1}]],
            batch_size=1,
            temperature=0.0,
            parse_fn=lambda x: x,
            raise_on_error=True,
        )
