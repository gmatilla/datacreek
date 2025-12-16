"""LLM-based curation helpers using Langchain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from langchain.llms.base import BaseLLM
    from langchain.prompts import PromptTemplate

try:  # pragma: no cover - optional dependency
    import openai
except Exception:  # pragma: no cover - openai missing
    openai = None  # type: ignore

__all__ = ["propose_merge_split", "record_feedback", "fine_tune_from_feedback"]

_DEFAULT_PROMPT = (
    "You are a curation assistant. Given a list of entities, "
    "suggest which should be merged or split. "
    "Return JSON with keys 'merge' and 'split'.\nEntities:\n{entities}"
)


def propose_merge_split(
    entities: Iterable[str], *, llm: "BaseLLM", prompt: str | None = None
) -> Dict[str, Any]:
    """Return merge/split suggestions for ``entities`` using ``llm``.

    Parameters
    ----------
    entities:
        Iterable of textual descriptions, one per entity.
    llm:
        Any Langchain-compatible language model.
    prompt:
        Optional custom prompt template with a ``{entities}`` placeholder.
    """
    try:
        from langchain.prompts import PromptTemplate

        text = "\n".join(entities)
        template = PromptTemplate.from_template(prompt or _DEFAULT_PROMPT)
        response = llm.predict(template.format(entities=text))
    except Exception:  # pragma: no cover - optional dependency missing
        text = "\n".join(entities)
        tmpl = (prompt or _DEFAULT_PROMPT).format(entities=text)
        response = llm.predict(tmpl)
    try:
        return json.loads(response)
    except Exception as err:
        raise ValueError("invalid LLM response") from err


def record_feedback(
    suggestion: Dict[str, Any], accepted: bool, path: str | Path
) -> None:
    """Append ``suggestion`` with ``accepted`` flag to ``path`` as JSONL."""
    entry = {"suggestion": suggestion, "accepted": accepted}
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def fine_tune_from_feedback(path: str | Path, *, model: str) -> Any:
    """Launch a fine-tuning job from feedback records."""
    if openai is None:
        raise RuntimeError("openai package required for fine-tuning")
    return openai.FineTuningJob.create(training_file=str(path), model=model)
