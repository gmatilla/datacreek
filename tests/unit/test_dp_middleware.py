import sys
import types

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import Response
from starlette.testclient import TestClient

import datacreek.security.dp_middleware as dp


class DummyEntry:
    def __init__(self, used, max_):
        self.epsilon_used = used
        self.epsilon_max = max_


class DummySession:
    def __init__(self, entry):
        self.entry = entry
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def get(self, model, tid):
        return self.entry

    def commit(self):
        self.committed = True


def make_client(entry, monkeypatch):
    session = DummySession(entry)
    dbmod = types.SimpleNamespace(SessionLocal=lambda: session, TenantPrivacy=object)
    monkeypatch.setitem(sys.modules, "datacreek.db", dbmod)
    # Ensure the datacreek package exposes the stub module
    import datacreek

    monkeypatch.setattr(datacreek, "db", dbmod, raising=False)

    def dummy_compute(vals, alphas=None):
        return vals[0]

    # Patch both the middleware module and the public DP module
    monkeypatch.setattr(dp, "compute_epsilon", dummy_compute)
    monkeypatch.setattr("datacreek.dp.compute_epsilon", dummy_compute)
    app = Starlette(middleware=[Middleware(dp.DPBudgetMiddleware)])

    @app.route("/")
    async def index(request):
        return Response("ok")

    return TestClient(app), session


def test_no_headers():
    app = Starlette(middleware=[Middleware(dp.DPBudgetMiddleware)])

    @app.route("/")
    async def index(request):
        return Response("hi")

    with TestClient(app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.text == "hi"
        assert "X-Epsilon-Remaining" not in resp.headers


def test_invalid_headers(monkeypatch):
    client, _ = make_client(None, monkeypatch)
    resp = client.get(
        "/",
        headers={
            dp.DPBudgetMiddleware.header_tenant: "x",
            dp.DPBudgetMiddleware.header_epsilon: "y",
        },
    )
    assert resp.status_code == 400


def test_tenant_missing(monkeypatch):
    client, session = make_client(None, monkeypatch)
    resp = client.get(
        "/",
        headers={
            dp.DPBudgetMiddleware.header_tenant: "1",
            dp.DPBudgetMiddleware.header_epsilon: "0.5",
        },
    )
    assert resp.status_code == 403
    assert resp.headers.get("X-Epsilon-Remaining") == "0.000000"
    assert not session.committed


def test_budget_exceeded(monkeypatch):
    entry = DummyEntry(1.0, 1.5)
    client, session = make_client(entry, monkeypatch)
    resp = client.get(
        "/",
        headers={
            dp.DPBudgetMiddleware.header_tenant: "1",
            dp.DPBudgetMiddleware.header_epsilon: "1.0",
        },
    )
    assert resp.status_code == 403
    assert resp.headers.get("X-Epsilon-Remaining") == "0.000000"
    assert entry.epsilon_used == 1.0
    assert not session.committed


def test_budget_updated(monkeypatch):
    entry = DummyEntry(0.2, 1.0)
    client, session = make_client(entry, monkeypatch)
    resp = client.get(
        "/",
        headers={
            dp.DPBudgetMiddleware.header_tenant: "1",
            dp.DPBudgetMiddleware.header_epsilon: "0.3",
        },
    )
    assert resp.status_code == 200
    # epsilon_used updated and remaining reflected
    assert entry.epsilon_used == 0.5
    assert session.committed
    assert resp.headers["X-Epsilon-Remaining"] == "0.500000"
