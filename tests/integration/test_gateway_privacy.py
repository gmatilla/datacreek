import importlib
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi import Depends, FastAPI, Header
from fastapi.testclient import TestClient


def setup_module(module):
    db_path = "test_dp_gateway.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    if "datacreek.db" in sys.modules:
        del sys.modules["datacreek.db"]
    import datacreek.db as db

    importlib.reload(db)
    db.init_db()
    db.Base.metadata.create_all(bind=db.engine)
    from hashlib import sha256

    def hash_key(k: str) -> str:
        return sha256(k.encode()).hexdigest()

    with db.SessionLocal() as session:
        user = db.User(username="alice", api_key=hash_key("key"), password_hash="pw")
        session.add(user)
        session.commit()
        uid = user.id
        from datacreek.security.tenant_privacy import set_tenant_limit

        set_tenant_limit(session, uid, epsilon_max=1.0)

    from datacreek.security.dp_middleware import DPBudgetMiddleware

    def get_db():
        db_s = db.SessionLocal()
        try:
            yield db_s
        finally:
            db_s.close()

    app = FastAPI()
    app.add_middleware(DPBudgetMiddleware)

    @app.post("/dp/sample")
    def sample(_=Depends(get_db)):
        return {"ok": True}

    @app.get("/dp/budget")
    def budget_route(tenant: str = Header("0", alias="X-Tenant"), _=Depends(get_db)):
        from datacreek.security.tenant_privacy import get_budget

        with db.SessionLocal() as s:
            info = get_budget(s, int(tenant))
        return info or {
            "epsilon_max": 0.0,
            "epsilon_used": 0.0,
            "epsilon_remaining": 0.0,
        }

    module.client = TestClient(app)
    module.api_key = "key"
    module.user_id = uid


def test_budget_allows_within_limit():
    headers = {"X-API-Key": api_key, "X-Tenant": str(user_id), "X-Epsilon": "0.6"}
    res = client.post("/dp/sample", headers=headers)
    assert res.status_code == 200
    assert pytest.approx(0.4) == float(res.headers["X-Epsilon-Remaining"])


def test_budget_rejects_excess():
    headers = {"X-API-Key": api_key, "X-Tenant": str(user_id), "X-Epsilon": "0.6"}
    res = client.post("/dp/sample", headers=headers)
    assert res.status_code == 403
    assert res.headers["X-Epsilon-Remaining"] == "0.000000"


def test_budget_reset():
    from datacreek.db import SessionLocal
    from datacreek.security.tenant_privacy import reset_all

    with SessionLocal() as session:
        reset_all(session)
    headers = {"X-API-Key": api_key, "X-Tenant": str(user_id), "X-Epsilon": "0.5"}
    res = client.post("/dp/sample", headers=headers)
    assert res.status_code == 200
    assert pytest.approx(0.5) == float(res.headers["X-Epsilon-Remaining"])


def test_get_budget_endpoint():
    """Verify that the user can query their remaining budget."""

    from datacreek.db import SessionLocal
    from datacreek.security.tenant_privacy import reset_all

    with SessionLocal() as session:
        reset_all(session)

    headers = {
        "X-API-Key": api_key,
        "X-Tenant": str(user_id),
        "X-Epsilon": "0.1",
    }
    res = client.post("/dp/sample", headers=headers)
    assert res.status_code == 200

    res = client.get(
        "/dp/budget",
        headers={"X-API-Key": api_key, "X-Tenant": str(user_id)},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["epsilon_max"] == pytest.approx(1.0)
    assert data["epsilon_used"] == pytest.approx(0.1)
    assert data["epsilon_remaining"] == pytest.approx(0.9)


def test_budget_exceed_logs(monkeypatch, tmp_path, caplog):
    """Tenant over budget should emit JSON log and 403 response."""

    db_path = tmp_path / "dp_exceed.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    if "datacreek.db" in sys.modules:
        importlib.reload(sys.modules["datacreek.db"])
    else:
        importlib.import_module("datacreek.db")
    import datacreek.db as db

    db.init_db()
    from datacreek.security.tenant_privacy import set_tenant_limit

    with db.SessionLocal() as session:
        set_tenant_limit(session, 7, epsilon_max=3.0)

    from datacreek.security.dp_middleware import DPBudgetMiddleware

    app = FastAPI()
    app.add_middleware(DPBudgetMiddleware)

    @app.post("/x")
    def _sample():
        return {"ok": True}

    client_local = TestClient(app)
    caplog.set_level(logging.INFO, logger="datacreek.privacy")

    res = client_local.post("/x", headers={"X-Tenant": "7", "X-Epsilon": "5"})
    assert res.status_code == 403
    assert res.headers["X-Epsilon-Remaining"] == "0.000000"
    assert any('"allowed": false' in r.message for r in caplog.records)
