"""Safety filter for incoming text, blocking toxic/NSFW payloads."""

from __future__ import annotations

import re
from typing import Optional

try:  # optional dependency
    from transformers import pipeline
except Exception:  # pragma: no cover - dependency missing
    pipeline = None  # type: ignore

try:  # optional Prometheus metric
    from prometheus_client import CollectorRegistry, Counter
except Exception:  # pragma: no cover - metrics disabled
    CollectorRegistry = Counter = None  # type: ignore

# Tiny HF toxicity model
_MODEL_ID = "unitary/toxic-roberta"

# Regex heuristics for NSFW content (simple demo set)
_NSFW_RE = re.compile(r"\b(?:porn|xxx|sex)\b", re.I)

# Prometheus counter for blocked payloads
INGEST_TOXIC_BLOCKS: Optional[Counter]
if Counter is not None:
    _REGISTRY = CollectorRegistry(auto_describe=True)
    INGEST_TOXIC_BLOCKS = Counter(
        "ingest_toxic_blocks_total",
        "Total number of toxic payloads blocked during ingestion",
        registry=_REGISTRY,
    )
else:  # pragma: no cover - metrics disabled
    INGEST_TOXIC_BLOCKS = None

# Lazily loaded classification pipeline
_TOXICITY_PIPE = None


def _load_pipeline() -> None:
    """Load the Hugging Face pipeline if not already loaded."""
    global _TOXICITY_PIPE
    if _TOXICITY_PIPE is None and pipeline is not None:
        _TOXICITY_PIPE = pipeline("text-classification", model=_MODEL_ID)


def _toxicity_score(text: str) -> float:
    """Return probability of toxicity from the model if available."""
    _load_pipeline()
    if _TOXICITY_PIPE is None:  # pragma: no cover - model missing
        return 0.0
    try:
        res = _TOXICITY_PIPE(text)[0]
    except Exception:  # pragma: no cover - inference failure
        return 0.0
    label = res.get("label", "").lower()
    score = float(res.get("score", 0.0))
    # Some models use LABEL_1 for toxic class
    if "toxic" in label or label == "label_1":
        return score
    return 1 - score


def filter_text(text: str, threshold: float = 0.7) -> Optional[str]:
    """Return text if safe, otherwise ``None``.

    The global score :math:`s` combines model toxicity :math:`s_{tox}` and
    regex heuristic :math:`s_{nsfw}` using

    .. math::

        s = 0.5 (s_{tox} + s_{nsfw})

    Blocking occurs when ``s > threshold`` (default 0.7). The Prometheus
    counter ``ingest_toxic_blocks_total`` is incremented when a text is blocked.
    """
    s_tox = _toxicity_score(text)
    s_nsfw = 1.0 if _NSFW_RE.search(text) else 0.0
    s = 0.5 * (s_tox + s_nsfw)
    if s > threshold:
        if INGEST_TOXIC_BLOCKS is not None:  # pragma: no branch - metric optional
            INGEST_TOXIC_BLOCKS.inc()
        return None
    return text
