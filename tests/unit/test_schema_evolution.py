import os

import sqlalchemy as sa

from datacreek.utils.schema_evolution import add_column_if_missing


def test_add_column_if_missing(tmp_path):
    db = tmp_path / "test.db"
    engine = sa.create_engine(f"sqlite:///{db}")
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE demo(id INTEGER PRIMARY KEY, a TEXT)"))

    add_column_if_missing(engine, "demo", "b", "INTEGER")
    cols = [c["name"] for c in sa.inspect(engine).get_columns("demo")]
    assert "b" in cols

    # second call should not raise or add duplicate
    add_column_if_missing(engine, "demo", "b", "INTEGER")
    cols2 = [c["name"] for c in sa.inspect(engine).get_columns("demo")]
    assert cols2.count("b") == 1
