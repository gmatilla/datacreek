import json
from pathlib import Path

import datacreek.utils.delta_export as de


def test_export_delta_list(tmp_path):
    data = [{"a": 1}, {"b": 2}]
    path = de.export_delta(data, root=str(tmp_path), org_id=1, kind="foo")
    assert path.name == "data.jsonl"
    content = path.read_text().splitlines()
    assert [json.loads(c) for c in content] == data


def test_export_delta_string(tmp_path):
    path = de.export_delta("text", root=str(tmp_path), org_id="x", kind="bar")
    assert path.read_text() == "text"


def test_lakefs_commit(monkeypatch):
    def fake_run(cmd, check, capture_output=False, text=False):
        assert capture_output and text
        return type(
            "R",
            (),
            {
                "stdout": json.dumps({"id": "sha1"}),
                "returncode": 0,
            },
        )()

    monkeypatch.setattr(de.subprocess, "run", fake_run)
    sha = de.lakefs_commit(Path("/tmp/file"), repo="r")
    assert sha == "sha1"


def test_lakefs_commit_error(monkeypatch):
    def bad_run(*a, **k):
        raise RuntimeError

    monkeypatch.setattr(de.subprocess, "run", bad_run)
    assert de.lakefs_commit(Path("/tmp/file"), repo="r") is None


def test_delta_optimize(monkeypatch):
    called = {}

    def fake_run(cmd, check):
        called["cmd"] = cmd
        called["check"] = check

    monkeypatch.setattr(de.subprocess, "run", fake_run)
    de.delta_optimize(Path("/data"))
    assert called["cmd"] == ["delta", "optimize", "/data", "--zorder-by", "org_id,kind"]


def test_delta_vacuum(monkeypatch):
    called = {}

    def fake_run(cmd, check):
        called["cmd"] = cmd
        called["check"] = check

    monkeypatch.setattr(de.subprocess, "run", fake_run)
    de.delta_vacuum(Path("/data"), retain_days=30)
    assert called["cmd"] == ["delta", "vacuum", "/data", "--retain", "30"]
