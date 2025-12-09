"""Task detection and dataset formatting utilities."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def detect_task(dataset: Any) -> str:
    """Infer the training task for a dataset.

    The function first checks for an explicit ``task`` entry in ``dataset.info``
    metadata. If absent, heuristics based on column names are applied following
    the project specification:

    * columns ``chosen`` and ``rejected``  -> ``"rlhf_dpo"``
    * column ``answer``                    -> ``"qa"``
    * column ``label`` or ``labels``      -> ``"classification"``
    * otherwise                           -> ``"generation"``

    Parameters
    ----------
    dataset:
        Object representing a dataset. It must expose ``info`` with a
        ``metadata`` mapping and ``column_names`` listing available fields.

    Returns
    -------
    str
        Detected task label.
    """

    meta = getattr(getattr(dataset, "info", None), "metadata", {}) or {}
    task = meta.get("task")
    if task:
        return str(task)

    columns = set(getattr(dataset, "column_names", []))
    if {"chosen", "rejected"}.issubset(columns):
        return "rlhf_dpo"
    if "answer" in columns:
        return "qa"
    if "label" in columns or "labels" in columns:
        return "classification"
    return "generation"


def _join_prompt_response(prompt: str, response: str, eos_token: str) -> str:
    """Concatenate ``prompt`` and ``response`` with an EOS token."""

    return f"{prompt}\n{response}{eos_token}"


DEFAULT_EOS = "<eos>"


def format_sft(
    sample: Mapping[str, Any], eos_token: Optional[str] = None
) -> Dict[str, str]:
    """Format a sample for supervised fine-tuning or QA generation."""

    prompt = sample.get("prompt") or sample.get("question") or ""
    response = sample.get("response") or sample.get("answer") or ""
    eos = DEFAULT_EOS if eos_token is None else eos_token
    text = _join_prompt_response(prompt, response, eos)
    return {"prompt": prompt, "text": text}


def format_classif(
    sample: Mapping[str, Any], eos_token: Optional[str] = None
) -> Dict[str, Any]:
    """Format a sample for classification training."""

    prompt = sample.get("text") or sample.get("prompt") or ""
    label = sample.get("label") or sample.get("labels")
    eos = DEFAULT_EOS if eos_token is None else eos_token
    text = _join_prompt_response(prompt, str(label), eos)
    return {"prompt": prompt, "labels": label, "text": text}


def format_rlhf(
    sample: Mapping[str, Any], eos_token: Optional[str] = None
) -> Dict[str, str]:
    """Format a sample for preference-based RLHF training (e.g., DPO)."""

    prompt = sample.get("prompt") or ""
    eos = DEFAULT_EOS if eos_token is None else eos_token
    chosen = f"{sample.get('chosen', '')}{eos}"
    rejected = f"{sample.get('rejected', '')}{eos}"
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
