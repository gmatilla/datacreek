import os
import types
from pathlib import Path

import datacreek.backends as backends


class DummyRedis:
    def __init__(self, **kw):
        self.kw = kw

    def ping(self):
        return True


class DummyGraph:
    def __init__(self, name, client):
        self.name = name
        self.client = client


class DummySession:
    def __init__(self):
        self.calls = []

    def run(self, stmt):
        self.calls.append(stmt)
        if "fail" in stmt:
            raise RuntimeError("boom")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class DummyDriver:
    def __init__(self, session):
        self.session_obj = session

    def session(self):
        return self.session_obj


def test_get_redis_client_env(monkeypatch):
    backends.get_redis_client.cache_clear()
    monkeypatch.setattr(backends, "redis", types.SimpleNamespace(Redis=DummyRedis))
    monkeypatch.setattr(
        backends, "get_redis_config", lambda cfg: {"host": "h", "port": 6380}
    )
    monkeypatch.setattr(backends, "load_config_with_overrides", lambda p: {})
    monkeypatch.setenv("REDIS_HOST", "envh")
    monkeypatch.setenv("REDIS_PORT", "1234")
    client = backends.get_redis_client()
    assert isinstance(client, DummyRedis)
    assert client.kw["host"] == "envh"
    assert client.kw["port"] == 1234


def test_get_neo4j_driver_with_migrations(monkeypatch, tmp_path):
    backends.get_neo4j_driver.cache_clear()
    session = DummySession()
    driver = DummyDriver(session)
    monkeypatch.setattr(
        backends,
        "GraphDatabase",
        types.SimpleNamespace(driver=lambda uri, auth: driver),
    )
    monkeypatch.setattr(
        backends, "ensure_neo4j_indexes", lambda d: session.run("index")
    )
    files = []
    monkeypatch.setattr(backends, "run_cypher_file", lambda d, f: files.append(f))
    monkeypatch.setattr(
        backends, "get_neo4j_config", lambda cfg: {"run_migrations": True}
    )
    monkeypatch.setattr(backends, "load_config_with_overrides", lambda p: {})
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost")
    monkeypatch.setenv("NEO4J_USER", "u")
    monkeypatch.setenv("NEO4J_PASSWORD", "p")
    result = backends.get_neo4j_driver()
    assert result is driver
    assert "2025-07-haa_index.cypher" in files


def test_ensure_neo4j_indexes(monkeypatch):
    session = DummySession()
    driver = DummyDriver(session)
    backends.ensure_neo4j_indexes(driver)
    assert len(session.calls) >= 8


def test_run_cypher_file(tmp_path):
    session = DummySession()
    driver = DummyDriver(session)
    path = (
        Path(backends.__file__).resolve().parent.parent / "migrations" / "temp.cypher"
    )
    path.write_text("A;B;")
    try:
        backends.run_cypher_file(driver, "temp.cypher")
    finally:
        path.unlink()
    assert session.calls == ["A", "B"]


def test_get_redis_graph(monkeypatch):
    backends.get_redis_graph.cache_clear()
    monkeypatch.setattr(backends, "RedisGraph", DummyGraph)
    monkeypatch.setattr(backends, "redis", types.SimpleNamespace(Redis=DummyRedis))
    monkeypatch.setattr(
        backends, "get_redis_config", lambda cfg: {"host": "h", "port": 1}
    )
    monkeypatch.setattr(backends, "load_config_with_overrides", lambda p: {})
    monkeypatch.setenv("USE_REDIS_GRAPH", "1")
    g = backends.get_redis_graph("name")
    assert isinstance(g, DummyGraph)
    assert g.name == "name"


def test_get_redis_graph_disabled(monkeypatch):
    backends.get_redis_graph.cache_clear()
    monkeypatch.setenv("USE_REDIS_GRAPH", "0")
    assert backends.get_redis_graph("n") is None


def test_get_s3_storage(monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "b")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
    monkeypatch.setenv("S3_PREFIX", "p")
    calls = {}

    class Boto:
        def client(self, *_args, **kw):
            calls.update(kw)
            return "cli"

    import sys

    boto = Boto()
    monkeypatch.setitem(sys.modules, "boto3", boto)
    monkeypatch.setattr(backends, "boto3", boto)
    store = backends.get_s3_storage()
    assert store.bucket == "b"
    assert store.prefix == "p/"
    assert calls.get("aws_access_key_id") == "k"


def test_get_s3_storage_no_bucket(monkeypatch):
    monkeypatch.delenv("S3_BUCKET", raising=False)
    assert backends.get_s3_storage() is None
