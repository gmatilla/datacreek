import importlib.abc
import importlib.util
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    "flyway_dry_run",
    Path(__file__).resolve().parents[1] / "scripts" / "flyway_dry_run.py",
)
flyway_dry_run = importlib.util.module_from_spec(spec)
assert isinstance(spec.loader, importlib.abc.Loader)
spec.loader.exec_module(flyway_dry_run)


class DummySession:
    def __init__(self):
        self.queries = []

    def run(self, q):
        self.queries.append(q)
        return SimpleNamespace(single=lambda: {"c": 0, "u": 0})

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class DummyDriver:
    def __init__(self):
        self.session_obj = DummySession()

    def session(self):
        return self.session_obj


def test_flyway_dry_run_executes_all_migrations(tmp_path, monkeypatch):
    monkeypatch.setattr(flyway_dry_run, "MIGR_DIR", tmp_path, raising=False)
    (tmp_path / "001.cypher").write_text("CREATE INDEX x ON :A(id);")
    (tmp_path / "002.cypher").write_text("CREATE INDEX y ON :B(id);")
    driver = DummyDriver()
    before, after = flyway_dry_run.dry_run(driver)
    assert before == "0-0"
    assert after == "0-0"
    # two snapshot queries plus two migration statements
    assert len(driver.session_obj.queries) == 4
