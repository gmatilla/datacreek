"""Language detection gating helper.

The module uses a fastText language identification model when available,
falling back to ``langdetect``. A Prometheus counter ``lang_skipped_total``
tracks how many payloads were skipped because their language was not among the
allowed set.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

try:  # optional dependencies
    import fasttext
except Exception:  # pragma: no cover - fastText missing
    fasttext = None  # type: ignore

try:  # fallback language detector
    from langdetect import detect
except Exception:  # pragma: no cover - langdetect missing
    detect = None  # type: ignore

try:  # optional Prometheus metric
    from prometheus_client import CollectorRegistry, Counter
except Exception:  # pragma: no cover - metrics disabled
    CollectorRegistry = Counter = None  # type: ignore

# Prometheus counter for skipped payloads
LANG_SKIPPED_TOTAL: Optional[Counter]
if Counter is not None:
    _REGISTRY = CollectorRegistry(auto_describe=True)
    LANG_SKIPPED_TOTAL = Counter(
        "lang_skipped_total",
        "Total number of payloads skipped due to unsupported language",
        registry=_REGISTRY,
    )
else:  # pragma: no cover - metrics disabled
    LANG_SKIPPED_TOTAL = None

# Lazy loaded fastText model
_FT_MODEL = None


def _load_model() -> None:
    """Load the fastText model from ``FASTTEXT_LID_PATH`` if available."""
    global _FT_MODEL
    if _FT_MODEL is None and fasttext is not None:
        model_path = os.getenv("FASTTEXT_LID_PATH", "")
        if model_path:
            try:
                _FT_MODEL = fasttext.load_model(model_path)
            except Exception:  # pragma: no cover - model loading failure
                _FT_MODEL = None


def detect_language(text: str) -> str:
    """Return ISO language code for ``text``.

    fastText is used when both the library and model file are present; otherwise
    ``langdetect`` is used as a lightweight fallback.
    """
    _load_model()
    if _FT_MODEL is not None:
        try:
            label = _FT_MODEL.predict(text, k=1)[0][0]
            return label.split("__label__")[-1]
        except Exception:  # pragma: no cover - prediction failure
            pass
    if detect is not None:
        try:
            return detect(text)
        except Exception:  # pragma: no cover - detection failure
            return "unknown"
    return "unknown"


def should_process(text: str, allowed: Iterable[str] = ("fr", "en")) -> bool:
    """Return ``True`` if detected language is in ``allowed``.

    When the language is not permitted the ``lang_skipped_total`` counter is
    incremented and ``False`` is returned to indicate that further processing
    (e.g. Whisper/BLIP) should be skipped.
    """
    lang = detect_language(text)
    if lang not in set(allowed):
        if LANG_SKIPPED_TOTAL is not None:  # pragma: no branch - metric optional
            LANG_SKIPPED_TOTAL.inc()
        return False
    return True
