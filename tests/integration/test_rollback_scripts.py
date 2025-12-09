import importlib.util
import os
import subprocess
from pathlib import Path


class DummySession:
    def __init__(self):
        self.queries = []

    def run(self, q):
        self.queries.append(q)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class DummyDriver:
    def __init__(self):
        self.session_obj = DummySession()

    def session(self):
        return self.session_obj


def test_backup_datastores(tmp_path):
    lmdb_src = tmp_path / "lmdb_src"
    lmdb_src.mkdir()
    (lmdb_src / "data.mdb").write_text("x")
    out = tmp_path / "out"
    env = os.environ.copy()
    env["LMDB_PATH"] = str(lmdb_src)
    env["REDIS_BACKUP_CMD"] = "touch"
    subprocess.check_call(["scripts/backup_datastores.sh", str(out)], env=env)
    assert (out / "lmdb_backup").exists()
    assert (out / "redis_dump.rdb").exists()


def test_revert_cache(tmp_path):
    backup = tmp_path / "back"
    lmdb_dst = tmp_path / "lmdb_dst"
    (backup / "lmdb_backup").mkdir(parents=True)
    (backup / "redis_dump.rdb").touch()
    env = os.environ.copy()
    env["LMDB_PATH"] = str(lmdb_dst)
    env["REDIS_RESTORE_CMD"] = "cat"  # no-op
    subprocess.check_call(["scripts/revert_cache.sh", str(backup)], env=env)
    assert lmdb_dst.exists()


def test_revert_ingestion_contains_restart():
    data = Path("scripts/revert_ingestion.sh").read_text()
    assert "docker compose restart ingestion" in data


def test_rollback_haa_index_executes_queries(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "rollback_haa", Path("scripts/rollback_haa_index.py")
    )
    mod = importlib.util.module_from_spec(spec)
    assert isinstance(spec.loader, importlib.abc.Loader)
    spec.loader.exec_module(mod)

    driver = DummyDriver()
    monkeypatch.setattr(mod, "get_neo4j_driver", lambda: driver)
    mod.main()
    assert driver.session_obj.queries == [
        "DROP CONSTRAINT haa_pair_unique IF EXISTS",
        "DROP INDEX haa_pair IF EXISTS",
    ]
