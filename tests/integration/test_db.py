import importlib
import os
import sys

from sqlalchemy import inspect


def test_init_db_creates_tables(tmp_path, monkeypatch):
    db_path = tmp_path / "db.sqlite"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    if "datacreek.db" in sys.modules:
        importlib.reload(sys.modules["datacreek.db"])
    else:
        importlib.import_module("datacreek.db")
    import datacreek.db as db

    db.init_db()
    insp = inspect(db.engine)
    tables = set(insp.get_table_names())
    assert {"users", "sources", "datasets", "tenant_privacy"}.issubset(tables)
    cols = [c["name"] for c in insp.get_columns("users")]
    assert "password_hash" in cols
    src_cols = [c["name"] for c in insp.get_columns("sources")]
    assert "entities" in src_cols
    assert "facts" in src_cols
