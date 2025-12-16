"""Language detection gating helper.

Uses fastText when available and falls back to :mod:`langdetect`.
Skipped payloads are counted via the shared Prometheus registry so the
``lang_skipped_total`` metric is visible alongside the other `datacreek.analysis`
gauges.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable

try:
    import fasttext
except Exception:  # pragma: no cover - optional dependency
    fasttext = None  # type: ignore

try:
    from langdetect import detect
except Exception:  # pragma: no cover - fallback missing
    detect = None  # type: ignore

try:
    from datacreek.analysis import monitoring
except Exception:  # pragma: no cover - metrics optional
    monitoring = None  # type: ignore

LOGGER = logging.getLogger(__name__)

_FT_MODEL: fasttext.FastText | None = None
_LANG_SKIPPED = 0
_FASTTEXT_WARNED = False


def _load_model() -> None:
    """Load the fastText model if a path is configured."""

    global _FT_MODEL, _FASTTEXT_WARNED
    if _FT_MODEL is not None:
        return
    if fasttext is None:
        if not _FASTTEXT_WARNED:
            LOGGER.warning("fastText not installed; language detection falls back to langdetect")
            _FASTTEXT_WARNED = True
        return
    model_path = os.getenv("FASTTEXT_LID_PATH", "")
    if not model_path:
        if not _FASTTEXT_WARNED:
            LOGGER.warning("FASTTEXT_LID_PATH not configured; language detection uses langdetect")
            _FASTTEXT_WARNED = True
        return
    try:
        _FT_MODEL = fasttext.load_model(model_path)
    except Exception:
        LOGGER.warning("Failed to load fastText model at %s; falling back to langdetect", model_path)
        _FT_MODEL = None


def detect_language(text: str) -> str:
    """Return the ISO language code for ``text``."""

    _load_model()
    if _FT_MODEL is not None:
        try:
            label = _FT_MODEL.predict(text, k=1)[0][0]
            return label.split("__label__")[-1]
        except Exception:
            pass
    if detect is not None:
        try:
            return detect(text)
        except Exception:
            return "unknown"
    return "unknown"


def should_process(text: str, allowed: Iterable[str] = ("fr", "en")) -> bool:
    """Return ``True`` if the detected language belongs to ``allowed``."""

    global _LANG_SKIPPED
    lang = detect_language(text)
    if lang not in set(allowed):
        _LANG_SKIPPED += 1
        if monitoring is not None:
            monitoring.update_metric("lang_skipped_total", float(_LANG_SKIPPED))
        return False
    return True
