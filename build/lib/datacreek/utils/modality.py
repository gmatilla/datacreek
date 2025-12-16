from __future__ import annotations

"""Utilities for detecting speech or writing modality."""

import os
from functools import lru_cache
from pathlib import Path

_SPOKEN_MARKERS = {
    "um",
    "uh",
    "er",
    "erm",
    "you know",
    "like",
}


@lru_cache(maxsize=1024)
def detect_modality(resource: str) -> str:
    """Return modality for ``resource``.

    The function accepts either a file path or textual content. When ``resource``
    points to an existing file, the modality is inferred from the file extension
    as ``TEXT``, ``IMAGE``, ``AUDIO`` or ``CODE``. Otherwise the content is
    analysed and ``"spoken"`` or ``"written"`` is returned using a lightweight
    heuristic. This dual behaviour preserves backwards compatibility with
    earlier tests.
    """

    path = Path(resource)
    if path.exists() and path.is_file():
        ext = path.suffix.lower()
        if ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp"}:
            return "IMAGE"
        if ext in {".mp3", ".wav", ".flac", ".m4a", ".ogg"}:
            return "AUDIO"
        if ext in {".py", ".js", ".c", ".cpp", ".go", ".rs", ".java", ".html", ".css"}:
            return "CODE"
        return "TEXT"

    txt = resource.lower()
    for w in _SPOKEN_MARKERS:
        if w in txt:
            return "spoken"
    punct = sum(txt.count(p) for p in ".!?")
    words = len(txt.split())
    if words and punct / words < 0.05:
        return "spoken"
    return "written"
