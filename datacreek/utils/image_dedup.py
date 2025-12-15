"""Perceptual hash deduplication for images."""

from __future__ import annotations

import hashlib
import logging
import os
from threading import Lock
from typing import Iterable

try:  # pragma: no cover - optional dependencies
    import imagehash
    from PIL import Image
except Exception:  # pragma: no cover - missing libs
    imagehash = None  # type: ignore
    Image = None  # type: ignore

LOGGER = logging.getLogger(__name__)

# Bloom filter defaults (1G bits ≈ 128MB, 7 hash functions)
_DEFAULT_BITS = 1_073_741_824
_DEFAULT_HASHES = 7

_FILTER_BITS = int(os.getenv("IMAGE_DEDUP_BITS", _DEFAULT_BITS))
_K = int(os.getenv("IMAGE_DEDUP_HASHES", _DEFAULT_HASHES))
_LOCK = Lock()
_FILTER = bytearray(_FILTER_BITS // 8)
FILTER = _FILTER


def configure_bloom_filter(*, bits: int | None = None, hashes: int | None = None) -> None:
    """Reinitialize the bloom filter with new size/hash counts."""

    global _FILTER_BITS, _K, _FILTER
    with _LOCK:
        if bits is not None:
            _FILTER_BITS = bits
        if hashes is not None:
            _K = hashes
        _FILTER = bytearray(_FILTER_BITS // 8)


def _hashes(data: bytes, *, bits: int, count: int) -> Iterable[int]:
    """Yield ``count`` bloom filter indices for ``data``."""
    digest = hashlib.blake2b(data, digest_size=32).digest()
    for i in range(count):
        chunk = int.from_bytes(digest[i * 4 : (i + 1) * 4], "little")
        yield chunk % bits


def check_duplicate(path: str) -> bool:
    """Return ``True`` if ``path`` is likely a duplicate.

    The perceptual hash is computed with :func:`imagehash.phash`. Hash bits are
    combined inside the Bloom filter. False positives are possible (~0.01%).
    """

    if imagehash is None or Image is None:
        LOGGER.warning("image dedup requires Pillow + imagehash; skipping for %s", path)
        return False

    try:
        img = Image.open(path)
    except Exception as exc:
        LOGGER.exception("Cannot open image %s", path)
        return False

    try:
        ph = imagehash.phash(img)
    finally:
        img.close()

    data = ph.hash.tobytes()
    seen = True
    with _LOCK:
        for h in _hashes(data, bits=_FILTER_BITS, count=_K):
            idx = h // 8
            bit = 1 << (h % 8)
            if not _FILTER[idx] & bit:
                seen = False
        for h in _hashes(data, bits=_FILTER_BITS, count=_K):
            idx = h // 8
            bit = 1 << (h % 8)
            _FILTER[idx] |= bit
    return seen
