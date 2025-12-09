import fakeredis

from datacreek import tasks


class DummyDataset:
    def __init__(self):
        self.events = []

    def _record_event(self, event, msg, **kwargs):
        self.events.append((event, msg, kwargs))


def test_update_status_and_record_error():
    r = fakeredis.FakeRedis()
    key = "task:1"
    tasks._update_status(r, key, tasks.TaskStatus.INGESTING, 0.1)
    assert r.hget(key, "status") == b"ingesting"
    assert float(r.hget(key, "progress")) == 0.1
    import json

    history = [json.loads(x) for x in r.lrange(f"{key}:history", 0, -1)]
    assert history[0]["status"] == "ingesting"

    ds = DummyDataset()
    try:
        raise ValueError("boom")
    except Exception as exc:
        tasks._record_error(r, key, exc, ds)

    assert b"boom" in r.hget(key, "error")
    assert ds.events[0][0] == "task_error"
