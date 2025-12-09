"""FastAPI microservice performing pre-ingestion safety, language and audio checks.

The service is designed to run as a side-car in front of the Kafka ingestion
pipeline.  It exposes a single ``POST /filter`` endpoint that validates a
payload using three stages:

1. **Language identification** – payloads outside of French or English are
   skipped and accounted by the ``lang_skipped_total`` Prometheus counter.  The
   detection is deliberately lightweight for unit testing.
2. **Audio signal-to-noise gate** – the caller provides the current audio SNR
   and a history of previous SNR measurements.  The threshold is computed as

   .. math::

       \text{thr}_{SNR} = 6 + 0.5\,\sigma_{SNR}

   where :math:`\sigma_{SNR}` is the standard deviation of the history.  When
   the current SNR falls below this threshold the request is rejected with HTTP
   422 and both ``filter_block_total`` and ``snr_block_total`` are incremented
   to distinguish audio issues from toxicity blocks.
3. **Toxicity/NSFW filter** – delegates to :func:`ingest.safety_filter.filter_text`
   which combines a tiny transformer model and regex heuristics.  Failing the
   filter also results in HTTP 422 and a counter increment.

The service is meant to be deployed behind Envoy and returns HTTP 422 whenever
validation fails so upstream components can drop the payload before it reaches
Kafka.
"""

from __future__ import annotations

from math import sqrt
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .safety_filter import filter_text

try:  # optional Prometheus metrics
    from prometheus_client import CollectorRegistry, Counter
except Exception:  # pragma: no cover - metrics disabled
    CollectorRegistry = Counter = None  # type: ignore

# ---------------------------------------------------------------------------
# Prometheus counters
# ---------------------------------------------------------------------------
FILTER_BLOCK_TOTAL: Optional[Counter]
LANG_SKIPPED_TOTAL: Optional[Counter]
SNR_BLOCK_TOTAL: Optional[Counter]
if Counter is not None:
    _REGISTRY = CollectorRegistry(auto_describe=True)
    FILTER_BLOCK_TOTAL = Counter(
        "filter_block_total",
        "Payloads blocked by the ingest-filter side-car",
        registry=_REGISTRY,
    )
    LANG_SKIPPED_TOTAL = Counter(
        "lang_skipped_total",
        "Payloads skipped due to unsupported language",
        registry=_REGISTRY,
    )
    SNR_BLOCK_TOTAL = Counter(
        "snr_block_total",
        "Payloads rejected due to low signal-to-noise ratio",
        registry=_REGISTRY,
    )
else:  # pragma: no cover - metrics disabled
    FILTER_BLOCK_TOTAL = None
    LANG_SKIPPED_TOTAL = None
    SNR_BLOCK_TOTAL = None

# Languages allowed for downstream BLIP/Whisper processing
ALLOWED_LANGS = {"fr", "en"}


class Payload(BaseModel):
    """Input payload inspected by the ingest filter."""

    text: str = ""
    snr: float = 0.0
    snr_history: List[float] = []


def detect_language(text: str) -> str:
    """Return a crude language ID using simple heuristics.

    The implementation purposefully avoids external dependencies.  It merely
    checks for a handful of accented characters to classify French and defaults
    to English otherwise.
    """

    lower = text.lower()
    if any(ch in lower for ch in "àâçéèêëîïôùûüÿœ") or "bonjour" in lower:
        return "fr"
    if "hola" in lower:
        return "es"
    return "en"


def compute_snr_threshold(history: List[float]) -> float:
    """Compute :math:`\text{thr}_{SNR}` from a history of SNR samples."""

    if not history:
        return 6.0
    mean = sum(history) / len(history)
    sigma = sqrt(sum((x - mean) ** 2 for x in history) / len(history))
    return 6.0 + 0.5 * sigma


def create_app() -> FastAPI:
    """Return a configured FastAPI application for the ingest filter."""

    app = FastAPI()

    @app.post("/filter")
    def filter_payload(payload: Payload) -> dict:
        """Validate ``payload`` and return a status object.

        Rejections raise :class:`HTTPException` with status 422.  Successful
        payloads return ``{"status": "ok"}`` while unsupported languages result
        in ``{"status": "lang_skipped"}``.
        """

        lang = detect_language(payload.text)
        if lang not in ALLOWED_LANGS:
            if LANG_SKIPPED_TOTAL is not None:  # pragma: no branch - metric optional
                LANG_SKIPPED_TOTAL.inc()
            return {"status": "lang_skipped", "lang": lang}

        thr = compute_snr_threshold(payload.snr_history)
        if payload.snr < thr:
            if FILTER_BLOCK_TOTAL is not None:  # pragma: no branch - metric optional
                FILTER_BLOCK_TOTAL.inc()
            if SNR_BLOCK_TOTAL is not None:  # pragma: no branch - metric optional
                SNR_BLOCK_TOTAL.inc()
            raise HTTPException(status_code=422, detail="low_snr")

        if filter_text(payload.text) is None:
            if FILTER_BLOCK_TOTAL is not None:  # pragma: no branch - metric optional
                FILTER_BLOCK_TOTAL.inc()
            raise HTTPException(status_code=422, detail="blocked")

        return {"status": "ok"}

    return app
