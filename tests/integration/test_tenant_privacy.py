import importlib
import os
import sys

from sqlalchemy import inspect


def test_tenant_privacy_budget(tmp_path, monkeypatch):
    db_path = tmp_path / "budget.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    if "datacreek.db" in sys.modules:
        importlib.reload(sys.modules["datacreek.db"])
    else:
        importlib.import_module("datacreek.db")
    import datacreek.db as db

    db.init_db()
    from datacreek.security.tenant_privacy import can_consume_epsilon, set_tenant_limit

    insp = inspect(db.engine)
    assert "tenant_privacy" in insp.get_table_names()
    with db.SessionLocal() as session:
        set_tenant_limit(session, 1, epsilon_max=1.0)
        assert can_consume_epsilon(session, 1, 0.6) is True
        assert can_consume_epsilon(session, 1, 0.5) is False
