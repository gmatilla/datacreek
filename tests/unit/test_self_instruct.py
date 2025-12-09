import asyncio
import json
import types

import pytest

import datacreek.utils.self_instruct as si


def test_generate_self_instruct_direct(monkeypatch):
    monkeypatch.setattr(si, "validate_output", lambda t, o: True)
    out = si.generate_with_self_instruct(lambda x: "hello", "inst", template="t")
    assert out == "hello"


def test_generate_self_instruct_reformat(monkeypatch):
    calls = iter(["bad", "again"])
    monkeypatch.setattr(si, "validate_output", lambda t, o: o == "good")
    monkeypatch.setattr(
        si,
        "parse_qa_pairs",
        lambda o: (
            [types.SimpleNamespace(question="q", answer="a")] if o == "again" else []
        ),
    )
    monkeypatch.setattr(json, "dumps", lambda obj: "good")
    out = si.generate_with_self_instruct(
        lambda x: next(calls), "inst", template="t", retries=2
    )
    assert out == "good"


@pytest.mark.asyncio
async def test_generate_self_instruct_async(monkeypatch):
    seq = iter(["no", "good"])

    async def llm(_):
        return next(seq)

    monkeypatch.setattr(si, "validate_output", lambda t, o: o == "good")
    monkeypatch.setattr(si, "parse_qa_pairs", lambda o: [])
    out = await si.generate_with_self_instruct_async(
        llm, "inst", template="t", retries=2
    )
    assert out == "good"


def test_auto_tool_calls(monkeypatch):
    def ins(text, tools):
        return text + "!" if tools else text

    assert si.auto_tool_calls("hi", [("t", "ex")], ins) == "hi!"
    assert si.auto_tool_calls("hi", [("t", "ex")], lambda t, tt: "") == "hi"
    # None result should preserve original text
    assert si.auto_tool_calls("ok", [("a", "b")], lambda t, tt: None) == "ok"


def test_generate_self_instruct_failure(monkeypatch):
    monkeypatch.setattr(si, "validate_output", lambda t, o: False)
    monkeypatch.setattr(si, "parse_qa_pairs", lambda o: [])
    with pytest.raises(RuntimeError):
        si.generate_with_self_instruct(lambda x: "bad", "inst", template="t", retries=1)


@pytest.mark.asyncio
async def test_generate_self_instruct_async_failure(monkeypatch):
    async def llm(_):
        return "bad"

    monkeypatch.setattr(si, "validate_output", lambda t, o: False)
    monkeypatch.setattr(si, "parse_qa_pairs", lambda o: [])
    with pytest.raises(RuntimeError):
        await si.generate_with_self_instruct_async(llm, "inst", template="t", retries=1)


def test_generate_self_instruct_json_error(monkeypatch):
    """Validation failure when JSON serialization raises an exception."""
    monkeypatch.setattr(si, "validate_output", lambda t, o: False)
    monkeypatch.setattr(
        si,
        "parse_qa_pairs",
        lambda o: [types.SimpleNamespace(question="q", answer="a")],
    )
    monkeypatch.setattr(
        si.json, "dumps", lambda obj: (_ for _ in ()).throw(ValueError("boom"))
    )
    with pytest.raises(RuntimeError):
        si.generate_with_self_instruct(lambda x: "bad", "inst", template="t", retries=1)


@pytest.mark.asyncio
async def test_generate_self_instruct_async_json_error(monkeypatch):
    async def llm(_):
        return "bad"

    monkeypatch.setattr(si, "validate_output", lambda t, o: False)
    monkeypatch.setattr(
        si,
        "parse_qa_pairs",
        lambda o: [types.SimpleNamespace(question="q", answer="a")],
    )
    monkeypatch.setattr(
        si.json, "dumps", lambda obj: (_ for _ in ()).throw(ValueError("boom"))
    )
    with pytest.raises(RuntimeError):
        await si.generate_with_self_instruct_async(llm, "inst", template="t", retries=1)
