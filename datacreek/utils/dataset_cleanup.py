"""Utilities for post-processing generated datasets."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Dict, List


def deduplicate_pairs(
    pairs: List[Dict[str, Any]], threshold: float = 0.9
) -> List[Dict[str, Any]]:
    """Return ``pairs`` with near-duplicates removed.

    Examples are considered duplicates when both the question and answer are
    highly similar (ratio >= ``threshold``).
    """

    unique: List[Dict[str, Any]] = []
    for pair in pairs:
        q = pair.get("question", "")
        a = pair.get("answer", "")
        is_dup = False
        for u in unique:
            if (
                SequenceMatcher(None, q, u.get("question", "")).ratio() >= threshold
                and SequenceMatcher(None, a, u.get("answer", "")).ratio() >= threshold
            ):
                is_dup = True
                break
        if not is_dup:
            unique.append(pair)
    return unique
