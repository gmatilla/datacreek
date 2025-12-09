import asyncio

import pytest

from datacreek.models import llm_service


class DummyClient:
    def __init__(self):
        self.calls = []

    def chat_completion(self, messages):
        self.calls.append(("sync", messages))
        return "one"

    def batch_completion(self, batches):
        self.calls.append(("batch", batches))
        return ["b1", "b2"]

    async def async_batch_completion(self, batches):
        self.calls.append(("abatch", batches))
        return ["a1", "a2"]


def test_llm_service_sync(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(llm_service, "LLMClient", lambda **_: dummy)
    svc = llm_service.LLMService()
    out = svc("hi")
    assert out == "one"
    assert dummy.calls == [("sync", [{"role": "user", "content": "hi"}])]


def test_llm_service_batch(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(llm_service, "LLMClient", lambda **_: dummy)
    svc = llm_service.LLMService()
    out = svc.batch(["a", "b"])
    assert out == ["b1", "b2"]
    assert dummy.calls == [
        (
            "batch",
            [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],
        )
    ]


@pytest.mark.asyncio
async def test_llm_service_abatch(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(llm_service, "LLMClient", lambda **_: dummy)
    svc = llm_service.LLMService()
    out = await svc.acomplete(["x", "y"])
    assert out == ["a1", "a2"]
    assert dummy.calls == [
        (
            "abatch",
            [[{"role": "user", "content": "x"}], [{"role": "user", "content": "y"}]],
        )
    ]
    dummy.calls.clear()
    out2 = await svc.abatch(["x2"])
    assert out2 == ["a1", "a2"]
    assert dummy.calls == [("abatch", [[{"role": "user", "content": "x2"}]])]
