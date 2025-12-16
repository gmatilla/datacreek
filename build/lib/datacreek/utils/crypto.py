from __future__ import annotations

"""Simple encryption utilities used for PII fields."""

import base64
from itertools import cycle
from typing import Any, Dict, Iterable


def xor_encrypt(text: str, key: str) -> str:
    """Return base64-encoded XOR encryption of ``text`` with ``key``."""
    data = text.encode()
    key_bytes = key.encode()
    out = bytes(b ^ k for b, k in zip(data, cycle(key_bytes)))
    return base64.urlsafe_b64encode(out).decode()


def xor_decrypt(token: str, key: str) -> str:
    """Decode ``token`` produced by :func:`xor_encrypt`."""
    data = base64.urlsafe_b64decode(token.encode())
    key_bytes = key.encode()
    out = bytes(b ^ k for b, k in zip(data, cycle(key_bytes)))
    return out.decode()


def encrypt_pii_fields(
    record: Dict[str, Any], key: str, fields: Iterable[str]
) -> Dict[str, Any]:
    """Encrypt selected fields of ``record`` in-place and return it."""
    for f in fields:
        if f in record and record[f] is not None:
            record[f] = xor_encrypt(str(record[f]), key)
    return record


def decrypt_pii_fields(
    record: Dict[str, Any], key: str, fields: Iterable[str]
) -> Dict[str, Any]:
    """Decrypt selected fields of ``record`` in-place and return it."""
    for f in fields:
        if f in record and record[f] is not None:
            record[f] = xor_decrypt(str(record[f]), key)
    return record
