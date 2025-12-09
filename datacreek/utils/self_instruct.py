"""Simple helpers implementing a Self-Instruct style loop."""

from __future__ import annotations

import json
import logging
from typing import Awaitable, Callable, Iterable

from ..templates.library import validate_output
from .llm_processing import parse_qa_pairs

logger = logging.getLogger(__name__)


def generate_with_self_instruct(
    llm_call: Callable[[str], str],
    instruction: str,
    *,
    template: str,
    retries: int = 3,
) -> str:
    """Generate text repeatedly until it validates against ``template``.

    The helper queries ``llm_call`` with ``instruction``. If the raw output
    fails validation it falls back to ``parse_qa_pairs`` to reformat the text and
    tries validating again. Up to ``retries`` attempts are made before raising an
    error.
    """

    for attempt in range(retries):
        raw = llm_call(instruction)
        if validate_output(template, raw):
            return raw
        pairs = parse_qa_pairs(raw)
        if pairs:
            try:
                serial = json.dumps([p.__dict__ for p in pairs])
                if validate_output(template, serial):
                    return serial
            except Exception:
                pass
        logger.debug("Validation failed on attempt %d", attempt + 1)
    raise RuntimeError("LLM output did not pass validation")


async def generate_with_self_instruct_async(
    llm_call: Callable[[str], Awaitable[str]],
    instruction: str,
    *,
    template: str,
    retries: int = 3,
) -> str:
    """Asynchronous variant of :func:`generate_with_self_instruct`.

    Parameters
    ----------
    llm_call:
        Awaitable callable returning a completion for ``instruction``.
    instruction:
        Prompt to send to the language model.
    template:
        Validation template name.
    retries:
        Number of attempts allowed before raising an error.
    """

    for attempt in range(retries):
        raw = await llm_call(instruction)
        if validate_output(template, raw):
            return raw
        pairs = parse_qa_pairs(raw)
        if pairs:
            try:
                serial = json.dumps([p.__dict__ for p in pairs])
                if validate_output(template, serial):
                    return serial
            except Exception:
                pass
        logger.debug("Validation failed on attempt %d", attempt + 1)
    raise RuntimeError("LLM output did not pass validation")


def auto_tool_calls(
    text: str,
    tools: Iterable[tuple[str, str]],
    insert_fn: Callable[[str, Iterable[tuple[str, str]]], str],
) -> str:
    """Insert tool-call examples in ``text``.

    ``insert_fn`` receives the original text and a list of ``(name, example)``
    tuples and should return the text augmented with tool-call markers. When it
    returns ``None`` or an empty string the original text is preserved.
    """
    updated = insert_fn(text, tools)
    if not updated:
        return text
    return updated
