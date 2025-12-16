from __future__ import annotations

"""Lightweight helpers to insert API call placeholders."""

import re
from typing import Callable, Dict, Iterable, Tuple


def insert_tool_calls(text: str, tools: Iterable[Tuple[str, str]]) -> str:
    """Return ``text`` with simple tool call placeholders inserted.

    Parameters
    ----------
    text:
        Input text where tool calls should be added.
    tools:
        Iterable of ``(name, pattern)`` pairs. For each pair, the first
        occurrence matching ``pattern`` triggers insertion of
        ``"[TOOL:name(args)]"`` at the match location, where ``args`` is the
        matched substring.
    """

    out = text
    for name, pattern in tools:
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            # skip invalid patterns
            continue

        def repl(match: re.Match) -> str:
            return f"[TOOL:{name}({match.group(0)})]"

        out = regex.sub(repl, out, count=1)
    return out


def execute_tool_calls(text: str, tools: Dict[str, Callable[[str], str]]) -> str:
    """Execute tool call placeholders in ``text`` and return updated output.

    Parameters
    ----------
    text:
        Input text containing ``[TOOL:name(args)]`` markers.
    tools:
        Mapping of tool ``name`` to a callable accepting the ``args`` string
        and returning the replacement text. Unknown tool names are ignored.

    Returns
    -------
    str
        ``text`` with all placeholders replaced by the function outputs. If
        a tool raises an exception its placeholder is left intact.
    """

    pattern = re.compile(r"\[TOOL:(\w+)\((.*?)\)\]")

    def repl(match: re.Match) -> str:
        name, arg = match.group(1), match.group(2)
        func = tools.get(name)
        if not func:
            return match.group(0)
        try:
            return str(func(arg))
        except Exception:
            return match.group(0)

    return pattern.sub(repl, text)


def generate_with_tools(
    llm_call: Callable[[str], str],
    prompt: str,
    tools: Dict[str, Callable[[str], str]],
    *,
    insert_patterns: Iterable[Tuple[str, str]] = (),
    score_fn: Callable[[str], float] | None = None,
    retries: int = 1,
) -> str:
    """Generate text using tool calls when it improves a quality score.

    The helper applies :func:`insert_tool_calls` to ``prompt`` using
    ``insert_patterns`` and queries ``llm_call``. If ``score_fn`` is provided
    the output is only kept when ``score_fn`` yields a higher value than the
    baseline prompt without tools. Placeholders are replaced by calling
    ``execute_tool_calls``.

    Parameters
    ----------
    llm_call:
        Function invoking a language model with a string prompt.
    prompt:
        Base prompt sent to the model.
    tools:
        Mapping of tool names to callables executed on placeholder content.
    insert_patterns:
        ``(name, regex)`` pairs used to insert tool calls in ``prompt``.
    score_fn:
        Function evaluating text quality. When ``None`` the first tool assisted
        output is returned.
    retries:
        Maximum attempts with the tool augmented prompt.

    Returns
    -------
    str
        Best model output according to ``score_fn`` or the tool assisted text
        if no scoring is provided.
    """

    baseline = llm_call(prompt)
    best = baseline
    best_score = score_fn(baseline) if score_fn else None

    tool_prompt = insert_tool_calls(prompt, insert_patterns)
    for _ in range(retries):
        raw = llm_call(tool_prompt)
        executed = execute_tool_calls(raw, tools)
        if score_fn is None:
            return executed
        score = score_fn(executed)
        if best_score is None or score > best_score:
            best = executed
            best_score = score

    return best
