import sys
import types

import pytest

from datacreek.generators.conversation_generator import ConversationGenerator
from datacreek.models.qa import QAPair
from datacreek.models.results import QAGenerationResult


class DummyLLM:
    def __init__(self):
        self.config = {}


def make_stub_result(num_pairs=1):
    pairs = [QAPair(question=f"q{i}", answer=f"a{i}") for i in range(num_pairs)]
    return QAGenerationResult(summary="sum", qa_pairs=pairs)


class StubQAGenerator:
    def __init__(self, *a, **k):
        pass

    def process_document(self, *a, **k):
        num = k.get("num_pairs", 1)
        return make_stub_result(num)

    async def process_document_async(self, *a, **k):
        num = k.get("num_pairs", 1)
        return make_stub_result(num)


def _patch_qagenerator(monkeypatch):
    stub_mod = types.SimpleNamespace(QAGenerator=StubQAGenerator)
    monkeypatch.setitem(sys.modules, "datacreek.generators.qa_generator", stub_mod)


def test_conversation_generator_sync(monkeypatch):
    _patch_qagenerator(monkeypatch)
    gen = ConversationGenerator(DummyLLM())
    res = gen.process_document("t", num_pairs=2)
    assert res.summary == "sum"
    assert len(res.conversations) == 2
    convo = res.conversations[0]["conversations"]
    assert convo[1]["content"] == "q0"
    assert convo[2]["content"] == "a0"


@pytest.mark.asyncio
async def test_conversation_generator_async(monkeypatch):
    _patch_qagenerator(monkeypatch)
    gen = ConversationGenerator(DummyLLM())
    res = await gen.process_document_async("t", num_pairs=1)
    assert res.conversations[0]["conversations"][2]["content"] == "a0"
