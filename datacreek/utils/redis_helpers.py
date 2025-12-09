"""Decode Redis hash entries and JSON values."""

from __future__ import annotations

import json
from typing import Any, Mapping


def decode_hash(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``data`` with bytes decoded and JSON values parsed."""
    result: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(k, bytes):
            k = k.decode()
        if isinstance(v, bytes):
            v = v.decode()
        try:
            result[k] = json.loads(v)
        except Exception:
            result[k] = v
    return result
