import pytest
from fastapi import FastAPI

from datacreek.telemetry import init_tracing


def test_init_tracing():
    app = FastAPI()
    init_tracing(app)
    try:
        from opentelemetry import trace  # type: ignore
    except Exception:
        pytest.skip("opentelemetry not installed")

    assert isinstance(trace.get_tracer_provider(), trace.TracerProvider)


def test_vector_search_tracing(monkeypatch):
    """Verify that the vector search endpoint emits a tracing span."""
    try:
        from opentelemetry import trace  # type: ignore
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )
    except Exception:  # pragma: no cover - optional dep
        pytest.skip("opentelemetry not installed")

    exporter = InMemorySpanExporter()

    import importlib

    vector_mod = importlib.import_module("datacreek.routers.vector_router")

    class DummyDataset:
        def search_hybrid(self, query: str, k: int = 5, node_type: str = "chunk"):
            return [1]

    app = FastAPI()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor().instrument_app(app)
    app.dependency_overrides[vector_mod.get_current_user] = lambda: type(
        "U", (), {"id": 1}
    )()
    vector_mod._load_dataset = lambda name, user: DummyDataset()
    app.include_router(vector_mod.router)

    from fastapi.testclient import TestClient

    client = TestClient(app)
    res = client.post(
        "/vector/search",
        headers={"X-API-Key": "dummy"},
        json={"dataset": "demo", "query": "graph", "k": 1},
    )
    assert res.status_code == 200

    span_names = [s.name for s in exporter.get_finished_spans()]
    assert any("/vector/search" in name for name in span_names)
