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
        if "RETURN count(r) AS c" in query:

            class _Rec(dict):
                def single(self):
                    return {"c": 2, "sum": 5}

            return _Rec()
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


def test_normalize_script(monkeypatch, capsys):
    backend_stub = ModuleType("datacreek.backends")
    backend_stub.last_driver = DummyDriver()
    backend_stub.get_neo4j_driver = lambda: backend_stub.last_driver
    monkeypatch.setitem(sys.modules, "datacreek.backends", backend_stub)

    spec = importlib.util.spec_from_file_location(
        "normalize", root / "scripts" / "normalize_haa_pairs.py"
    )
    normalize = importlib.util.module_from_spec(spec)
    assert isinstance(spec.loader, importlib.abc.Loader)
    spec.loader.exec_module(normalize)

    normalize.main()
    out = capsys.readouterr().out.strip()
    assert out == "2-5"
    queries = backend_stub.last_driver.session_obj.queries
    assert any("SET r.startNodeId" in q for q in queries)
    assert any("FOREACH" in q for q in queries)
