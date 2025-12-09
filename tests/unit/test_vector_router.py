import importlib
import json

import pytest
from fastapi import HTTPException

router = importlib.import_module("datacreek.routers.vector_router")


class DummyUser:
    def __init__(self, id):
        self.id = id


class DummyDS:
    def __init__(self):
        self.search_called = False

    def search_hybrid(self, query, k=5, node_type="chunk"):
        self.search_called = True
        return [1, 2, 3]


def test_get_current_user(monkeypatch):
    user = DummyUser(7)
    monkeypatch.setattr(router, "get_user_by_key", lambda db, key: user)
    out = router.get_current_user(api_key="token")
    assert out is user
    monkeypatch.setattr(router, "get_user_by_key", lambda db, key: None)
    with pytest.raises(HTTPException):
        router.get_current_user(api_key="bad")


def test_vector_search(monkeypatch):
    dummy = DummyDS()
    monkeypatch.setattr(router, "_load_dataset", lambda name, user: dummy)
    payload = router.VectorSearchRequest(dataset="demo", query="hello")
    resp = router.vector_search(payload, user=DummyUser(1))
    assert json.loads(resp.body.decode()) == [1, 2, 3]
    assert dummy.search_called
