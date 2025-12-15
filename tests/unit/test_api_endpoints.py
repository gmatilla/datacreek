"""Extra tests covering the FastAPI endpoints added in ``datacreek.api``."""

from __future__ import annotations

import types

import pytest
from fastapi.testclient import TestClient

import datacreek.api as api


class DummySession:
    """Minimal session supporting the ORM calls needed by the endpoints."""

    def __init__(self, user: types.SimpleNamespace | None = None):
        self.user = user
        self.filters: dict[str, str] = {}

    def __enter__(self):
        return self

    def __exit__(self, *_) -> bool:
        return False

    def close(self):
        pass

    def query(self, model):
        return self

    def filter_by(self, **kwargs):
        self.filters = kwargs
        return self

    def first(self):
        if self.user and self.filters.get("username") == self.user.username:
            return self.user
        return None

    def add(self, obj):
        pass

    def commit(self):
        pass

    def refresh(self, obj):
        pass


class DummyRedis:
    """Simplified redis client used by the dataset listing endpoint."""

    def __init__(self, members: dict[str, set[str | bytes]] | None = None):
        self.members = members or {}

    def exists(self, key: str) -> bool:
        return key in self.members

    def smembers(self, key: str):
        return self.members.get(key, set())


@pytest.fixture(autouse=True)
def _patch_cache(monkeypatch):
    """Ensure caching helpers never hit the real Redis during the unit tests."""

    monkeypatch.setattr(api, "_cache_user", lambda user: None)
    monkeypatch.setattr(api, "_cache_dataset", lambda ds: None)
    yield


def _session_with_user(monkeypatch, user: types.SimpleNamespace | None):
    session = DummySession(user)
    monkeypatch.setattr(api, "SessionLocal", lambda: session)
    return session


def test_login_rotates_api_key(monkeypatch):
    user = types.SimpleNamespace(username="demo", password_hash="hash", id=1, api_key="old")
    _session_with_user(monkeypatch, user)
    monkeypatch.setattr(api, "verify_password", lambda u, pwd: pwd == "secret")
    monkeypatch.setattr(api, "_rotate_api_key", lambda db, u: "new-key")

    client = TestClient(api.app)
    resp = client.post("/api/login", json={"username": "demo", "password": "secret"})

    assert resp.status_code == 200
    assert resp.json()["api_key"] == "new-key"


def test_register_returns_api_key(monkeypatch):
    user = types.SimpleNamespace(username="new", api_key="api-hash", id=2)
    _session_with_user(monkeypatch, None)
    monkeypatch.setattr(
        api,
        "create_user_with_generated_key",
        lambda db, username, password=None, **_: (user, "plain-key"),
    )

    client = TestClient(api.app)
    resp = client.post("/api/register", json={"username": "new", "password": "pass"})

    assert resp.status_code == 201
    assert resp.json()["api_key"] == "plain-key"
    assert resp.json()["username"] == "new"


def test_session_endpoint_reports_user(monkeypatch):
    user = types.SimpleNamespace(username="demo", id=1)
    _session_with_user(monkeypatch, user)
    monkeypatch.setattr(api, "get_user_by_key", lambda db, key: user)

    client = TestClient(api.app)
    resp = client.get("/api/session", headers={"X-API-Key": "token"})

    assert resp.status_code == 200
    assert resp.json()["username"] == "demo"


def test_list_datasets_fetches_user_sets(monkeypatch):
    user = types.SimpleNamespace(username="demo", id=1)
    _session_with_user(monkeypatch, user)
    monkeypatch.setattr(api, "get_user_by_key", lambda db, key: user)
    redis = DummyRedis({"user:1:datasets": {b"beta", b"alpha"}})
    monkeypatch.setattr(api, "get_redis_client", lambda: redis)

    client = TestClient(api.app)
    resp = client.get("/api/datasets", headers={"X-API-Key": "token"})

    assert resp.status_code == 200
    assert resp.json() == sorted(["alpha", "beta"])
