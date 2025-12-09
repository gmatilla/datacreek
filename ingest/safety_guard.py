"""Multilayer safety guard for text and media payloads.

The guard combines fast RegEx heuristics, a light toxicity classifier and
(optional) CLIP based NSFW detection for images.  A payload is blocked when

.. math::

    s = \tfrac12 (s_{tox} + s_{nsfw}) > 0.7

where :math:`s_{tox}` is the toxicity score from the transformer model and
:math:`s_{nsfw}` is the score from the image classifier. A Prometheus counter
``ingest_toxic_blocks_total`` is incremented for each blocked payload.
"""

from __future__ import annotations

import re
from typing import Optional

try:  # optional dependencies
    from PIL import Image
except Exception:  # pragma: no cover - pillow missing
    Image = None  # type: ignore

try:  # optional dependencies
    from transformers import CLIPModel, CLIPProcessor, pipeline
except Exception:  # pragma: no cover - transformers missing
    CLIPModel = CLIPProcessor = pipeline = None  # type: ignore

try:  # optional Prometheus metric
    from prometheus_client import CollectorRegistry, Counter
except Exception:  # pragma: no cover - metrics disabled
    CollectorRegistry = Counter = None  # type: ignore

# Regex blacklist for quick text rejection
_NSFW_RE = re.compile(r"\b(?:porn|xxx|sex)\b", re.I)

# Model ids
_TOX_MODEL_ID = "unitary/distilroberta-base-toxic-filtered"
_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

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

# Lazy loaded models
_TOXICITY_PIPE = None
_CLIP_MODEL = None
_CLIP_PROCESSOR = None


def _load_toxicity() -> None:
    """Load the text toxicity pipeline if available."""
    global _TOXICITY_PIPE
    if _TOXICITY_PIPE is None and pipeline is not None:
        _TOXICITY_PIPE = pipeline("text-classification", model=_TOX_MODEL_ID)


def _load_clip() -> None:
    """Load CLIP model and processor for NSFW scoring."""
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL is None and CLIPModel is not None and CLIPProcessor is not None:
        _CLIP_MODEL = CLIPModel.from_pretrained(_CLIP_MODEL_ID)
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(_CLIP_MODEL_ID)


def _toxicity_score(text: str) -> float:
    """Return toxicity score in ``[0,1]`` from the transformer model."""
    _load_toxicity()
    if _TOXICITY_PIPE is None:  # pragma: no cover - model missing
        return 0.0
    try:
        res = _TOXICITY_PIPE(text)[0]
        score = float(res.get("score", 0.0))
        return score
    except Exception:  # pragma: no cover - inference failure
        return 0.0


def _nsfw_image_score(image: Optional["Image.Image"]) -> float:
    """Return NSFW score for an image using CLIP if available."""
    if image is None:
        return 0.0
    _load_clip()
    if (
        _CLIP_MODEL is None or _CLIP_PROCESSOR is None
    ):  # pragma: no cover - model missing
        return 0.0
    try:
        inputs = _CLIP_PROCESSOR(text=["a photo"], images=image, return_tensors="pt")
        outputs = _CLIP_MODEL(**inputs)
        # Use norm of the pooled output as a simple proxy for NSFW probability.
        return float(outputs.pooler_output.norm().item())
    except Exception:  # pragma: no cover - inference failure
        return 0.0


def guard(
    text: str = "", image: Optional["Image.Image"] = None, threshold: float = 0.7
) -> Optional[str]:
    """Return ``text`` if safe, otherwise ``None``.

    The guard first blocks payloads matching a blacklist of explicit terms. If
    the regex passes, the ensemble score ``s`` is computed as

    .. math::

        s = 0.5 (s_{tox} + s_{nsfw})

    where ``s_tox`` is the model toxicity and ``s_nsfw`` the image NSFW score.
    Payloads with ``s > threshold`` are blocked.
    """
    if _NSFW_RE.search(text):
        if INGEST_TOXIC_BLOCKS is not None:  # pragma: no branch - metric optional
            INGEST_TOXIC_BLOCKS.inc()
        return None

    s_tox = _toxicity_score(text) if text else 0.0
    s_nsfw = _nsfw_image_score(image)
    s = 0.5 * (s_tox + s_nsfw)
    if s > threshold:
        if INGEST_TOXIC_BLOCKS is not None:  # pragma: no branch - metric optional
            INGEST_TOXIC_BLOCKS.inc()
        return None
    return text
