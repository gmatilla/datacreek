import importlib.abc
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

root = Path(__file__).resolve().parents[1]


class DummySession:
    def __init__(self):
        self.queries = []

    def run(self, query):
        self.queries.append(query)
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class DummyDriver:
    def __init__(self):
        self.session_obj = DummySession()

    def session(self):
        return self.session_obj


def test_dedup_script_calls_backup(monkeypatch):
    backend_stub = ModuleType("datacreek.backends")
    backend_stub.last_driver = DummyDriver()
    backend_stub.get_neo4j_driver = lambda: backend_stub.last_driver
    monkeypatch.setitem(sys.modules, "datacreek.backends", backend_stub)

    spec = importlib.util.spec_from_file_location(
        "dedup", root / "scripts" / "dedup_haa_relations.py"
    )
    dedup = importlib.util.module_from_spec(spec)
    assert isinstance(spec.loader, importlib.abc.Loader)
    spec.loader.exec_module(dedup)

    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)

    monkeypatch.setattr(dedup.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["dedup_haa_relations.py"])
    dedup.main()
    assert any("copy" in c for c in calls[0])
    q = backend_stub.last_driver.session_obj.queries[0]
    assert "DELETE" in q
