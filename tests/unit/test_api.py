import json
import types

import pytest
from fastapi import HTTPException

import datacreek.api as api


class DummyRedis:
    def __init__(self, data=None):
        self.data = data or {}

    def smembers(self, key):
        return self.data.get(key, set())

    def hgetall(self, key):
        return self.data.get(key, {})

    def hget(self, key, field):
        return self.data.get((key, field))

    def lrange(self, key, start, end):
        return self.data.get(key, [])

    def get(self, key):
        return self.data.get(key)


class DummyDataset:
    def __init__(self, name, owner_id=1, versions=None, redis_client=None):
        self.name = name
        self.owner_id = owner_id
        self.versions = versions or []
        self.redis_client = redis_client or DummyRedis()
        self.stage = "ready"

    @classmethod
    def from_redis(cls, client, key, driver):
        name = key.split(":", 1)[1]
        return cls(name, owner_id=1, redis_client=client)

    def delete_version(self, idx):
        if idx < 1 or idx > len(self.versions):
            raise IndexError
        self.versions.pop(idx - 1)


@pytest.fixture(autouse=True)
def patch_builder(monkeypatch):
    monkeypatch.setattr(api, "DatasetBuilder", DummyDataset)
    yield


def test_list_user_datasets_sorted(monkeypatch):
    user = types.SimpleNamespace(id=1)
    redis = DummyRedis({"user:1:datasets": {b"b", "a"}})
    monkeypatch.setattr(api, "get_redis_client", lambda: redis)
    assert api.list_user_datasets(current_user=user) == ["a", "b"]


def test_list_user_datasets_no_client(monkeypatch):
    user = types.SimpleNamespace(id=1)
    monkeypatch.setattr(api, "get_redis_client", lambda: None)
    assert api.list_user_datasets(current_user=user) == []


def test_load_dataset_owner_check(monkeypatch):
    user = types.SimpleNamespace(id=1)
    redis = DummyRedis()
    ds = DummyDataset("x", owner_id=2, redis_client=redis)
    monkeypatch.setattr(api, "get_redis_client", lambda: redis)
    monkeypatch.setattr(DummyDataset, "from_redis", lambda *a, **k: ds)
    with pytest.raises(HTTPException):
        api._load_dataset("x", user)


def test_load_dataset_no_client(monkeypatch):
    user = types.SimpleNamespace(id=1)
    monkeypatch.setattr(api, "get_redis_client", lambda: None)
    with pytest.raises(HTTPException):
        api._load_dataset("x", user)


def test_load_dataset_missing(monkeypatch):
    user = types.SimpleNamespace(id=1)
    redis = DummyRedis()
    monkeypatch.setattr(api, "get_redis_client", lambda: redis)

    def raise_key(*a, **k):
        raise KeyError

    monkeypatch.setattr(DummyDataset, "from_redis", raise_key)
    with pytest.raises(HTTPException):
        api._load_dataset("x", user)


def test_dataset_version_and_deletion(monkeypatch):
    redis = DummyRedis()
    ds = DummyDataset("d", versions=[{"v": 1}, {"v": 2}], redis_client=redis)
    monkeypatch.setattr(api, "get_redis_client", lambda: redis)
    monkeypatch.setattr(DummyDataset, "from_redis", lambda *a, **k: ds)
    user = types.SimpleNamespace(id=1)
    item = api.dataset_version_item("d", 1, current_user=user)
    assert item == {"v": 1}
    api.delete_dataset_version_item("d", 1, current_user=user)
    assert len(ds.versions) == 1


def test_dataset_version_item_missing(monkeypatch):
    redis = DummyRedis()
    ds = DummyDataset("d", versions=[{"v": 1}], redis_client=redis)
    monkeypatch.setattr(api, "get_redis_client", lambda: redis)
    monkeypatch.setattr(DummyDataset, "from_redis", lambda *a, **k: ds)
    user = types.SimpleNamespace(id=1)
    with pytest.raises(HTTPException):
        api.dataset_version_item("d", 5, current_user=user)


def test_dataset_version_item_low(monkeypatch):
    redis = DummyRedis()
    ds = DummyDataset("d", versions=[{"v": 1}], redis_client=redis)
    monkeypatch.setattr(api, "get_redis_client", lambda: redis)
    monkeypatch.setattr(DummyDataset, "from_redis", lambda *a, **k: ds)
    user = types.SimpleNamespace(id=1)
    with pytest.raises(HTTPException):
        api.dataset_version_item("d", 0, current_user=user)


def test_list_user_datasets_details(monkeypatch):
    redis = DummyRedis({"user:1:datasets": {"a"}, "dataset:a:progress": {"p": 1}})
    ds = DummyDataset("a", redis_client=redis)
    monkeypatch.setattr(api, "get_redis_client", lambda: redis)
    monkeypatch.setattr(DummyDataset, "from_redis", lambda *a, **k: ds)
    user = types.SimpleNamespace(id=1)
    res = api.list_user_datasets_details(current_user=user)
    assert res == [{"name": "a", "stage": "ready", "progress": {"p": 1}}]


def test_load_dataset_success(monkeypatch):
    redis = DummyRedis()
    ds = DummyDataset("a", owner_id=1, redis_client=redis)
    monkeypatch.setattr(api, "get_redis_client", lambda: redis)
    monkeypatch.setattr(DummyDataset, "from_redis", lambda *a, **k: ds)
    user = types.SimpleNamespace(id=1)
    result = api._load_dataset("a", user)
    assert result is ds and result.redis_client is redis


def test_dataset_export_result(monkeypatch):
    data = {
        "dataset:name:progress": None,
        ("dataset:name:progress", "export"): json.dumps({"key": "k"}),
        "k": "DATA",
    }
    redis = DummyRedis(data)
    ds = DummyDataset("name", redis_client=redis)
    monkeypatch.setattr(api, "get_redis_client", lambda: redis)
    monkeypatch.setattr(DummyDataset, "from_redis", lambda *a, **k: ds)
    user = types.SimpleNamespace(id=1)
    resp = api.dataset_export_result("name", api.ExportFormat.JSONL, current_user=user)
    assert "DATA" in resp.body.decode()


def test_dataset_export_not_found(monkeypatch):
    redis = DummyRedis()
    ds = DummyDataset("name", redis_client=redis)
    monkeypatch.setattr(api, "get_redis_client", lambda: redis)
    monkeypatch.setattr(DummyDataset, "from_redis", lambda *a, **k: ds)
    user = types.SimpleNamespace(id=1)
    with pytest.raises(HTTPException):
        api.dataset_export_result("name", api.ExportFormat.JSONL, current_user=user)


def test_dataset_progress_and_history(monkeypatch):
    data = {
        "dataset:n:progress": {"a": "1"},
        "dataset:n:progress:history": [b'{"x":1}', "invalid"],
    }
    redis = DummyRedis(data)
    ds = DummyDataset("n", redis_client=redis)
    monkeypatch.setattr(api, "get_redis_client", lambda: redis)
    monkeypatch.setattr(DummyDataset, "from_redis", lambda *a, **k: ds)
    user = types.SimpleNamespace(id=1)
    prog = api.dataset_progress("n", current_user=user)
    assert prog == {"a": 1}
    hist = api.dataset_progress_history("n", current_user=user)
    assert hist == [{"x": 1}]
