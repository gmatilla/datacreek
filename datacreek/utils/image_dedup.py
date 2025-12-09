"""Perceptual hash deduplication for images."""

from __future__ import annotations

import hashlib
from typing import Iterable

import imagehash
from PIL import Image

# Bloom filter parameters
M = 1_073_741_824  # 1G bits (~128MB)
K = 7
_BYTES = M // 8

# Global Bloom bit array
FILTER = bytearray(_BYTES)


def _hashes(data: bytes) -> Iterable[int]:
    """Yield ``K`` bloom filter indices for ``data``."""
    digest = hashlib.blake2b(data, digest_size=32).digest()
    for i in range(K):
        chunk = int.from_bytes(digest[i * 4 : (i + 1) * 4], "little")
        yield chunk % M


def check_duplicate(path: str) -> bool:
    """Return ``True`` if ``path`` is likely a duplicate.

    The perceptual hash is computed with :func:`imagehash.phash`. The hash
    bits are inserted into the Bloom filter and membership before insertion is
    returned. False positives are possible with probability around 0.01%.
    """
    img = Image.open(path)
    ph = imagehash.phash(img)
    img.close()
    data = ph.hash.tobytes()
    seen = True
    for h in _hashes(data):
        idx = h // 8
        bit = 1 << (h % 8)
        if not FILTER[idx] & bit:
            seen = False
    for h in _hashes(data):
        idx = h // 8
        bit = 1 << (h % 8)
        FILTER[idx] |= bit
    return seen
