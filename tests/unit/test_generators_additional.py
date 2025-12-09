import sys
import types

import pytest

from datacreek.generators.multi_tool_generator import MultiToolGenerator
from datacreek.generators.pref_generator import PrefListGenerator, PrefPairGenerator
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


def test_multi_tool_generator(monkeypatch):
    _patch_qagenerator(monkeypatch)
    gen = MultiToolGenerator(DummyLLM())
    res = gen.process_document("text", num_pairs=1)
    assert res.summary == "sum"
    conv = res.conversations[0]["conversations"]
    # tool calls inserted at positions 2 and 4
    assert conv[2]["tool_call"]["name"] == "search"
    assert conv[4]["tool_call"]["name"] == "calculator"


@pytest.mark.asyncio
async def test_multi_tool_async(monkeypatch):
    _patch_qagenerator(monkeypatch)
    gen = MultiToolGenerator(DummyLLM())
    res = await gen.process_document_async("text", num_pairs=1)
    assert res.summary == "sum"


def test_pref_pair_generator(monkeypatch):
    _patch_qagenerator(monkeypatch)
    gen = PrefPairGenerator(DummyLLM())
    res = gen.process_document("txt", num_pairs=1)
    assert len(res.pairs) == 1
    assert res.pairs[0]["chosen"] == "a0"


@pytest.mark.asyncio
async def test_pref_list_async(monkeypatch):
    _patch_qagenerator(monkeypatch)
    gen = PrefListGenerator(DummyLLM())
    res = await gen.process_document_async("txt", num_lists=1, list_size=1)
    assert res.responses[0]["answers"][0]["text"] == "a0"
