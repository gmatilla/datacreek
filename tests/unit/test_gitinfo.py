import types

import datacreek.utils.gitinfo as gi


def test_get_commit_hash(monkeypatch):
    monkeypatch.setattr(gi.subprocess, "check_output", lambda cmd, cwd: b"abc\n")
    assert gi.get_commit_hash() == "abc"


def test_get_commit_hash_error(monkeypatch):
    def fail(*a, **k):
        raise RuntimeError

    monkeypatch.setattr(gi.subprocess, "check_output", fail)
    assert gi.get_commit_hash() == "unknown"
