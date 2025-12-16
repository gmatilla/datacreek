"""Unified ingestion pipeline combining language gating and safety guard.

The function :func:`safe_ingest` first checks the language of the payload using
:mod:`datacreek.utils.lang_detect`.  If the language is not among the allowed
set, the payload is skipped and ``None`` is returned.  Otherwise, the text and
optional image are passed to :mod:`ingest.safety_guard` which blocks the payload
when the ensemble score exceeds the provided threshold.

Example
-------
>>> from ingest.pipeline import safe_ingest
>>> safe_ingest("bonjour")
'bonjour'

Notes
-----
Both submodules expose Prometheus counters:
``lang_skipped_total`` and ``ingest_toxic_blocks_total`` which are incremented
for skipped and blocked payloads respectively.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from PIL import Image

from datacreek.utils.lang_detect import should_process
from ingest.safety_guard import guard


def safe_ingest(
    text: str,
    allowed_langs: Iterable[str] = ("fr", "en"),
    *,
    image: Optional["Image.Image"] = None,
    threshold: float = 0.7,
) -> Optional[str]:
    """Return ``text`` if it passes language gating and safety checks.

    Parameters
    ----------
    text:
        Text payload to process.
    allowed_langs:
        Iterable of ISO codes; only these languages are processed. Others are
        skipped and counted via ``lang_skipped_total``.
    image:
        Optional ``PIL.Image`` associated with the text for NSFW scoring.
    threshold:
        Safety threshold :math:`s = 0.5 (s_{tox} + s_{nsfw})` above which the
        payload is blocked and ``ingest_toxic_blocks_total`` incremented.
    """
    if not should_process(text, allowed_langs):
        return None
    return guard(text, image=image, threshold=threshold)
