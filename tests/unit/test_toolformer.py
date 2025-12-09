import pytest

from datacreek.utils.toolformer import (
    execute_tool_calls,
    generate_with_tools,
    insert_tool_calls,
)


def test_insert_tool_calls_basic_and_invalid():
    text = "Search cats and dogs"
    result = insert_tool_calls(
        text,
        [
            ("search", r"cats"),
            ("filter", r"dogs"),
            ("bad", r"(unbalanced"),  # invalid pattern should be skipped
        ],
    )
    # Only first match replaced for each pattern
    assert result == "Search [TOOL:search(cats)] and [TOOL:filter(dogs)]"


def test_execute_tool_calls_unknown_and_error():
    def echo(x: str) -> str:
        return x.upper()

    def fail(x: str) -> str:
        raise RuntimeError()

    text = "[TOOL:echo(hi)] [TOOL:missing(x)] [TOOL:fail(oops)]"
    result = execute_tool_calls(text, {"echo": echo, "fail": fail})
    assert result == "HI [TOOL:missing(x)] [TOOL:fail(oops)]"


def test_generate_with_tools_scoring():
    baseline_result = "base"
    tool_prompt = "say [TOOL:echo(hi)]"
    calls = [tool_prompt, tool_prompt.replace("hi", "hello")]

    def llm_call(prompt: str) -> str:
        if prompt == "say hi":
            return baseline_result
        return calls.pop(0)

    result = generate_with_tools(
        llm_call,
        "say hi",
        {"echo": lambda x: x.upper()},
        insert_patterns=[("echo", r"hi")],
        score_fn=len,
        retries=2,
    )
    assert result == "say HELLO"


def test_generate_with_tools_no_score():
    result = generate_with_tools(
        lambda p: p,
        "hello",
        {"echo": lambda x: x.upper()},
        insert_patterns=[("echo", r"hello")],
    )
    assert result == "HELLO"
