import logging
import types

import datacreek.telemetry as telem


class DummyRecord(logging.LogRecord):
    def __init__(self):
        super().__init__("x", logging.INFO, __file__, 0, "msg", None, None)


def test_filter_no_trace(monkeypatch):
    monkeypatch.setattr(telem, "trace", None)
    record = DummyRecord()
    f = telem.TraceIdFilter()
    assert f.filter(record)
    assert record.trace_id == ""


def test_filter_with_trace(monkeypatch):
    class DummySpan:
        def __init__(self, tid):
            self._ctx = types.SimpleNamespace(trace_id=tid)

        def get_span_context(self):
            return self._ctx

    dummy_trace = types.SimpleNamespace(get_current_span=lambda: DummySpan(0x1234))
    monkeypatch.setattr(telem, "trace", dummy_trace)
    record = DummyRecord()
    assert telem.TraceIdFilter().filter(record)
    assert record.trace_id.endswith("1234")


def test_filter_span_none(monkeypatch):
    monkeypatch.setattr(
        telem, "trace", types.SimpleNamespace(get_current_span=lambda: None)
    )
    record = DummyRecord()
    assert telem.TraceIdFilter().filter(record)
    assert record.trace_id == ""


def test_init_tracing(monkeypatch):
    calls = {}

    class DummyApp:
        pass

    class Instr:
        def instrument_app(self, app):
            calls["app"] = app

    class Exporter:
        def __init__(self, endpoint):
            calls["endpoint"] = endpoint

    class Provider:
        def __init__(self, resource):
            calls["resource"] = resource

        def add_span_processor(self, proc):
            calls["processor"] = proc

    monkeypatch.setattr(
        telem,
        "trace",
        types.SimpleNamespace(
            set_tracer_provider=lambda p: calls.setdefault("provider", p),
            get_current_span=lambda: None,
        ),
    )
    monkeypatch.setattr(telem, "SpanExporter", Exporter)
    monkeypatch.setattr(telem, "FastAPIInstrumentor", lambda: Instr())
    monkeypatch.setattr(telem, "Resource", types.SimpleNamespace(create=lambda d: d))
    monkeypatch.setattr(telem, "TracerProvider", Provider)
    monkeypatch.setattr(
        telem, "BatchSpanProcessor", lambda exporter: ("proc", exporter)
    )
    monkeypatch.setattr(telem, "AioPikaInstrumentor", None)
    monkeypatch.setattr(telem, "SERVICE_NAME", "service_name", raising=False)

    logger = logging.getLogger()
    prev_filters = list(logger.filters)
    app = DummyApp()
    telem.init_tracing(app, service_name="svc", endpoint="http://x")
    assert calls["app"] is app
    assert calls["endpoint"] == "http://x"
    assert isinstance(logger.filters[-1], telem.TraceIdFilter)
    # restore filters
    logger.filters[:] = prev_filters


def test_init_tracing_aio(monkeypatch):
    """AioPika instrumentation is activated when available."""
    calls = {}

    class DummyApp:
        pass

    class Instr:
        def instrument_app(self, app):
            calls["app"] = app

    class AioInstr:
        def instrument(self):
            calls["aio"] = True

    monkeypatch.setattr(
        telem,
        "trace",
        types.SimpleNamespace(
            set_tracer_provider=lambda p: calls.setdefault("provider", p),
            get_current_span=lambda: None,
        ),
    )
    monkeypatch.setattr(telem, "SpanExporter", lambda endpoint=None: endpoint)
    monkeypatch.setattr(telem, "FastAPIInstrumentor", lambda: Instr())
    monkeypatch.setattr(telem, "Resource", types.SimpleNamespace(create=lambda d: d))

    class Provider:
        def __init__(self, resource):
            calls["resource"] = resource

        def add_span_processor(self, proc):
            calls["proc"] = proc

    monkeypatch.setattr(telem, "TracerProvider", Provider)
    monkeypatch.setattr(
        telem, "BatchSpanProcessor", lambda exporter: ("proc", exporter)
    )
    monkeypatch.setattr(telem, "AioPikaInstrumentor", lambda: AioInstr())
    monkeypatch.setattr(telem, "SERVICE_NAME", "service_name", raising=False)

    app = DummyApp()
    telem.init_tracing(app)
    assert calls["app"] is app
    assert calls.get("aio") is True


def test_init_tracing_noop(monkeypatch):
    """init_tracing should exit silently when dependencies are missing."""
    monkeypatch.setattr(telem, "trace", None)
    monkeypatch.setattr(telem, "FastAPIInstrumentor", None)
    monkeypatch.setattr(telem, "SpanExporter", None)
    telem.init_tracing(object())
