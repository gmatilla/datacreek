# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Text processing utilities
import json
import os
import re
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple

try:  # optional dependency
    import fasttext
except Exception:  # pragma: no cover - optional dependency missing
    fasttext = None  # type: ignore

FASTTEXT_BIN = os.getenv("FASTTEXT_MODEL", "lid.176.bin")
_FT_MODEL = None
_FASTTEXT_POOL: Queue | None = None

try:  # optional dependency
    from pint import UnitRegistry as _UnitRegistry
    from quantulum3 import parser as _qty_parser

    _PINT_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency missing
    _PINT_AVAILABLE = False

try:
    from unstructured.cleaners.core import clean as _clean

    _UNSTRUCTURED = True
except ImportError:  # pragma: no cover - optional dependency
    _UNSTRUCTURED = False

from .chunking import (
    contextual_chunk_split,
    semantic_chunk_split,
    sliding_window_chunks,
    summarized_chunk_split,
)


def split_into_chunks(
    text: str,
    chunk_size: int = 4000,
    overlap: int = 200,
    method: str | None = None,
    similarity_drop: float = 0.3,
) -> List[str]:
    """Split text into chunks using the configured method."""
    if method == "sliding":
        return sliding_window_chunks(text, chunk_size, overlap)
    if method == "semantic":
        return semantic_chunk_split(
            text, max_tokens=chunk_size, similarity_drop=similarity_drop
        )
    if method == "contextual":
        return contextual_chunk_split(text, max_tokens=chunk_size)
    if method == "summary":
        return summarized_chunk_split(text, max_tokens=chunk_size)

    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            sentences = current_chunk.split(". ")
            if len(sentences) > 3:
                current_chunk = ". ".join(sentences[-3:]) + "\n\n" + para
            else:
                current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def normalize_units(text: str) -> str:
    """Convert quantities in ``text`` to their SI representations."""

    if not _PINT_AVAILABLE:
        return text

    ureg = _UnitRegistry()
    try:
        quantities = _qty_parser.parse(text)
    except Exception:  # pragma: no cover - runtime issues
        return text

    offset = 0
    for q in quantities:
        if not hasattr(q, "span"):
            continue
        start, end = q.span
        try:
            qty = q.value * ureg(q.unit.name)
            qty_si = qty.to_base_units()
            replacement = f"{qty_si.magnitude:g} {qty_si.units}"
        except Exception:  # pragma: no cover - conversion failure
            continue
        text = text[: start + offset] + replacement + text[end + offset :]
        offset += len(replacement) - (end - start)
    return text


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON from text that might contain markdown or other content"""
    text = text.strip()

    # Try to parse as complete JSON
    if (
        text.startswith("{")
        and text.endswith("}")
        or text.startswith("[")
        and text.endswith("]")
    ):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Look for JSON within Markdown code blocks
    json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(json_pattern, text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try a more aggressive pattern
    json_pattern = r"\{[\s\S]*\}|\[[\s\S]*\]"
    match = re.search(json_pattern, text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not extract valid JSON from the response")


def clean_text(text: str) -> str:
    """Normalize ``text`` using ``unstructured`` when available."""

    if _UNSTRUCTURED:
        cleaned = _clean(text, extra_whitespace=True, dashes=True, bullets=True)
    else:
        # Fallback basic cleaning if ``unstructured`` isn't installed
        cleaned = re.sub(r"\s+", " ", text).strip()

    return normalize_units(cleaned)


def _get_ft_model(path: str) -> "fasttext.FastText":  # type: ignore[name-defined]
    """Return a fastText model instance from the global pool."""

    global _FT_MODEL, _FASTTEXT_POOL
    if _FASTTEXT_POOL is None:
        n = max(1, (os.cpu_count() or 1))
        q: Queue = Queue(maxsize=n)
        for _ in range(n):
            if _FT_MODEL is None:
                _FT_MODEL = fasttext.load_model(path)
            q.put(_FT_MODEL)
        _FASTTEXT_POOL = q
    model = _FASTTEXT_POOL.get()
    return model


def _release_ft_model(model: "fasttext.FastText") -> None:  # type: ignore[name-defined]
    """Return ``model`` to the global pool."""

    assert _FASTTEXT_POOL is not None
    _FASTTEXT_POOL.put(model)


def get_fasttext() -> "fasttext.FastText":  # type: ignore[name-defined]
    """Return a singleton fastText model for language detection."""
    if fasttext is None:  # pragma: no cover - optional dependency
        raise RuntimeError("fasttext is not installed")
    if not hasattr(get_fasttext, "_model"):
        get_fasttext._model = fasttext.load_model(FASTTEXT_BIN)
    return get_fasttext._model


def detect_language(
    text: str, model_path: str | None = None, *, return_prob: bool = False
) -> str | Tuple[str, float]:
    """Return ISO-639 code of ``text`` and optional probability."""

    if fasttext is None:
        return ("und", 0.0) if return_prob else "und"

    path = model_path or os.getenv("FASTTEXT_MODEL", "lid.176.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(f"fastText model not found: {path}")

    model = get_fasttext()
    labels, probs = model.predict(text.replace("\n", " "))

    if not labels:
        return ("und", 0.0) if return_prob else "und"

    lang = labels[0].replace("__label__", "")
    prob = float(probs[0]) if probs else 0.0
    return (lang, prob) if return_prob else lang
