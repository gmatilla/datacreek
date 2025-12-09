import sys
import types

import pytest

from datacreek.generators.tool_generator import ToolCallGenerator
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


def test_tool_generator(monkeypatch):
    _patch_qagenerator(monkeypatch)
    gen = ToolCallGenerator(DummyLLM())
    res = gen.process_document("text", num_pairs=1)
    conv = res.conversations[0]["conversations"]
    assert conv[2]["tool_call"]["name"] == "search"
    assert conv[3]["name"] == "search"


@pytest.mark.asyncio
async def test_tool_generator_async(monkeypatch):
    _patch_qagenerator(monkeypatch)
    gen = ToolCallGenerator(DummyLLM())
    res = await gen.process_document_async("text", num_pairs=1)
    assert res.summary == "sum"
