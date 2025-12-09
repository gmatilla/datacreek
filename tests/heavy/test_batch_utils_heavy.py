import asyncio

import pytest

import datacreek.utils.batch as batch


class DummyClient:
    def __init__(self):
        self.calls = []

    def batch_completion(self, messages, *, temperature=None, batch_size=None):
        self.calls.append((messages, temperature, batch_size))
        return ["resp:" + m[0]["content"] for m in messages]


class AsyncDummyClient:
    def __init__(self):
        self.calls = []

    async def async_batch_completion(
        self, messages, *, temperature=None, batch_size=None
    ):
        self.calls.append((messages, temperature, batch_size))
        return ["aresp:" + m[0]["content"] for m in messages]


@pytest.mark.heavy
def test_process_batches_success():
    client = DummyClient()
    messages = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
    res = batch.process_batches(
        client,
        messages,
        batch_size=1,
        temperature=0.1,
        parse_fn=lambda s: s.upper(),
    )
    assert res == ["RESP:A", "RESP:B"]
    assert len(client.calls) == 2


@pytest.mark.heavy
def test_process_batches_error_handling():
    class ErrClient(DummyClient):
        def batch_completion(self, *a, **k):
            raise RuntimeError("boom")

    # When raise_on_error is False, errors are logged and empty result returned
    res = batch.process_batches(
        ErrClient(),
        [[{"role": "user", "content": "a"}]],
        batch_size=1,
        temperature=0.0,
        parse_fn=lambda s: s,
        raise_on_error=False,
    )
    assert res == []

    with pytest.raises(RuntimeError):
        batch.process_batches(
            ErrClient(),
            [[{"role": "user", "content": "a"}]],
            batch_size=1,
            temperature=0.0,
            parse_fn=lambda s: s,
            raise_on_error=True,
        )


@pytest.mark.heavy
def test_async_process_batches_success():
    client = AsyncDummyClient()
    messages = [[{"role": "user", "content": "x"}], [{"role": "user", "content": "y"}]]

    async def run():
        res = await batch.async_process_batches(
            client,
            messages,
            batch_size=1,
            temperature=0.2,
            parse_fn=lambda s: s[::-1],
        )
        assert res == ["x:psera", "y:psera"]

    asyncio.run(run())
    assert len(client.calls) == 2


@pytest.mark.heavy
def test_async_process_batches_error_handling():
    class ErrAsyncClient(AsyncDummyClient):
        async def async_batch_completion(self, *a, **k):
            raise RuntimeError("boom")

    async def run_err():
        with pytest.raises(RuntimeError):
            await batch.async_process_batches(
                ErrAsyncClient(),
                [[{"role": "user", "content": "z"}]],
                batch_size=1,
                temperature=0.0,
                parse_fn=lambda s: s,
                raise_on_error=True,
            )

    asyncio.run(run_err())
