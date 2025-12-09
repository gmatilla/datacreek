"""Tests for the ``ingest-filter`` pre-ingestion microservice."""

from __future__ import annotations

from fastapi.testclient import TestClient

import ingest.ingest_filter as ingest_filter


def test_block_on_toxic_text(monkeypatch) -> None:
    """Toxic payloads are rejected with HTTP 422 and increment the counter."""

    def stub_filter(text: str, threshold: float = 0.7):  # pragma: no cover - stub
        return None

    monkeypatch.setattr(ingest_filter, "filter_text", stub_filter)
    app = ingest_filter.create_app()
    client = TestClient(app)

    start = (
        ingest_filter.FILTER_BLOCK_TOTAL._value.get()
        if ingest_filter.FILTER_BLOCK_TOTAL is not None
        else 0.0
    )
    snr_start = (
        ingest_filter.SNR_BLOCK_TOTAL._value.get()
        if ingest_filter.SNR_BLOCK_TOTAL is not None
        else 0.0
    )
    resp = client.post(
        "/filter",
        json={"text": "bad", "snr": 10.0, "snr_history": [10.0, 10.0, 10.0]},
    )
    assert resp.status_code == 422
    if ingest_filter.FILTER_BLOCK_TOTAL is not None:
        assert ingest_filter.FILTER_BLOCK_TOTAL._value.get() == start + 1
    if ingest_filter.SNR_BLOCK_TOTAL is not None:
        assert ingest_filter.SNR_BLOCK_TOTAL._value.get() == snr_start


def test_skip_on_unsupported_language() -> None:
    """Languages outside the allow-list are skipped and counted."""

    app = ingest_filter.create_app()
    client = TestClient(app)

    start = (
        ingest_filter.LANG_SKIPPED_TOTAL._value.get()
        if ingest_filter.LANG_SKIPPED_TOTAL is not None
        else 0.0
    )
    resp = client.post(
        "/filter", json={"text": "hola", "snr": 10.0, "snr_history": [10.0]}
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "lang_skipped"
    if ingest_filter.LANG_SKIPPED_TOTAL is not None:
        assert ingest_filter.LANG_SKIPPED_TOTAL._value.get() == start + 1


def test_block_on_low_snr() -> None:
    """Audio below the dynamic SNR threshold is rejected."""

    app = ingest_filter.create_app()
    client = TestClient(app)

    start = (
        ingest_filter.FILTER_BLOCK_TOTAL._value.get()
        if ingest_filter.FILTER_BLOCK_TOTAL is not None
        else 0.0
    )
    snr_start = (
        ingest_filter.SNR_BLOCK_TOTAL._value.get()
        if ingest_filter.SNR_BLOCK_TOTAL is not None
        else 0.0
    )
    resp = client.post(
        "/filter",
        json={"text": "hello", "snr": 5.0, "snr_history": [10.0, 8.0, 12.0]},
    )
    assert resp.status_code == 422
    if ingest_filter.FILTER_BLOCK_TOTAL is not None:
        assert ingest_filter.FILTER_BLOCK_TOTAL._value.get() == start + 1
    if ingest_filter.SNR_BLOCK_TOTAL is not None:
        assert ingest_filter.SNR_BLOCK_TOTAL._value.get() == snr_start + 1
