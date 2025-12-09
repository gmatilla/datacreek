import importlib
import sys

import fakeredis
import pytest

from datacreek.core.dataset import DatasetBuilder
from datacreek.pipelines import DatasetType


@pytest.fixture(autouse=True)
def reload_backpressure(monkeypatch):
    """Ensure backpressure module is reloaded with test limit."""
    if "datacreek.utils.backpressure" in sys.modules:
        importlib.reload(sys.modules["datacreek.utils.backpressure"])
    bp = importlib.import_module("datacreek.utils.backpressure")
    bp.set_limit(1)
    yield bp
    bp.set_limit(10000)


def test_dataset_ingest_task_queue_full(tmp_path, monkeypatch, reload_backpressure):
    bp = reload_backpressure
    client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.tasks.get_redis_client", lambda: client)
    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: None)
    import datacreek.tasks as tasks_mod

    tasks_mod.celery_app.conf.task_always_eager = True
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.to_redis(client, "dataset:demo")
    client.sadd("datasets", "demo")
    bp.acquire_slot()
    f = tmp_path / "doc.txt"
    f.write_text("x")
    with pytest.raises(RuntimeError):
        tasks_mod.dataset_ingest_task.delay("demo", str(f)).get()
    bp.release_slot()


def test_api_ingest_returns_429(tmp_path, monkeypatch, reload_backpressure):
    bp = reload_backpressure
    client = fakeredis.FakeStrictRedis()
    monkeypatch.setattr("datacreek.tasks.get_redis_client", lambda: client)
    monkeypatch.setattr("datacreek.tasks.get_neo4j_driver", lambda: None)
    monkeypatch.setattr("flask_login.login_required", lambda f: f)
    import datacreek.server.app as app_module

    app_module.REDIS = client
    import datacreek.tasks as tasks_mod

    tasks_mod.celery_app.conf.task_always_eager = True
    ds = DatasetBuilder(DatasetType.TEXT, name="demo")
    ds.redis_client = client
    ds.to_redis(client, "dataset:demo")
    app_module.DATASETS["demo"] = ds
    f = tmp_path / "doc.txt"
    f.write_text("hi")
    bp.acquire_slot()
    with app_module.app.test_client() as cl:
        cl.post("/api/login", json={"username": "alice", "password": "pw"})
        res = cl.post("/api/datasets/demo/ingest", json={"path": str(f)})
        assert res.status_code == 429
    bp.release_slot()
    app_module.DATASETS.clear()


def test_queue_fill_ratio_metric(monkeypatch, reload_backpressure):
    bp = reload_backpressure

    vals = []

    class DummyGauge:
        def set(self, v: float):
            vals.append(v)

    monkeypatch.setattr(
        "datacreek.analysis.monitoring.ingest_queue_fill_ratio",
        DummyGauge(),
        raising=False,
    )
    import datacreek.analysis.monitoring as mon

    monkeypatch.setitem(
        mon._METRICS, "ingest_queue_fill_ratio", mon.ingest_queue_fill_ratio
    )

    assert bp.acquire_slot()
    assert vals[-1] == pytest.approx(1.0)
    bp.release_slot()
    assert vals[-1] == pytest.approx(0.0)


def test_acquire_slot_with_backoff_spool(tmp_path, reload_backpressure):
    bp = reload_backpressure
    spool = tmp_path / "spool"
    bp.acquire_slot()
    data = {"name": "demo", "path": "x"}
    assert not bp.acquire_slot_with_backoff(
        1, 0.01, spool_dir=str(spool), spool_data=data
    )
    files = list(spool.iterdir())
    assert len(files) == 1
    import json

    assert json.loads(files[0].read_text()) == data
    bp.release_slot()


def test_acquire_slot_with_backoff_retries(monkeypatch, reload_backpressure):
    bp = reload_backpressure
    bp.acquire_slot()
    delays = []

    monkeypatch.setattr("time.sleep", lambda d: delays.append(d))
    assert not bp.acquire_slot_with_backoff(2, 0.01)
    assert delays == [0.01, 0.02, 0.04]
    bp.release_slot()


def test_backpressure_burst_drop_ratio(reload_backpressure):
    bp = reload_backpressure
    bp.set_limit(10)
    drops = 0
    ok = 0
    for _ in range(20):
        if bp.acquire_slot():
            ok += 1
            bp.release_slot()
        else:
            drops += 1
    total = ok + drops
    assert drops / total < 0.01
