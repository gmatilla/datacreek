from __future__ import annotations

import hashlib


def md5_file(path: str) -> str:
    """Return the MD5 checksum of the file at ``path``."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
