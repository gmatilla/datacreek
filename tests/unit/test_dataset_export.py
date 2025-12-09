from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from datacreek.utils import snapshot_template, snapshot_tokenizer


class DummyTokenizer:
    """Minimal tokenizer stub writing deterministic JSON."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def save_pretrained(self, path: str | Path) -> None:  # pragma: no cover - trivial
        out = Path(path) / "tokenizer.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(self.payload, fh, ensure_ascii=False, sort_keys=True)


def test_snapshot_tokenizer(tmp_path, monkeypatch):
    tok = DummyTokenizer({"hello": 1})

    # Record arguments to both the commit helper and LakeFS client
    called: dict = {}

    class FakeClient:
        def upload_object(self, repo, path, branch="main"):
            called["upload_repo"] = repo
            called["upload_path"] = Path(path)
            called["upload_branch"] = branch

    def fake_commit(path, repo):
        called["commit_path"] = Path(path)
        called["commit_repo"] = repo
        return "id"

    monkeypatch.setattr("datacreek.utils.dataset_export.lakefs_commit", fake_commit)

    path, digest1 = snapshot_tokenizer(
        tok, path=tmp_path, repo="foo", lakefs_client=FakeClient()
    )
    assert path == tmp_path / "tokenizer.json"
    data = path.read_bytes()
    assert digest1 == sha256(data).hexdigest()
    assert called["commit_repo"] == "foo" and called["commit_path"] == tmp_path
    assert called["upload_repo"] == "foo"
    assert called["upload_path"].name == "tokenizer.json"

    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["tokenizer_sha"] == digest1

    # Second snapshot should yield the same hash and overwrite metadata
    _, digest2 = snapshot_tokenizer(
        tok, path=tmp_path, repo="foo", lakefs_client=FakeClient()
    )
    assert digest2 == digest1
    meta2 = json.loads((tmp_path / "metadata.json").read_text())
    assert meta2["tokenizer_sha"] == digest1


def test_snapshot_template(tmp_path, monkeypatch):
    template = tmp_path / "prompt.jinja"
    template.write_text("Hello {{ name }}!\n", encoding="utf-8")

    called: dict = {}

    class FakeClient:
        def upload_object(self, repo, path, branch="main"):
            called["upload_repo"] = repo
            called["upload_path"] = Path(path)
            called["upload_branch"] = branch

    def fake_commit(path, repo):
        called["commit_path"] = Path(path)
        called["commit_repo"] = repo
        return "id"

    monkeypatch.setattr("datacreek.utils.dataset_export.lakefs_commit", fake_commit)

    out, digest = snapshot_template(
        template, path=tmp_path, repo="foo", lakefs_client=FakeClient()
    )
    assert out == tmp_path / "prompt.jinja"
    text = out.read_text(encoding="utf-8")
    assert text.splitlines()[0] == f"# sha256: {digest}"
    assert called["commit_repo"] == "foo" and called["commit_path"] == tmp_path
    assert called["upload_repo"] == "foo"
    assert called["upload_path"].name == "prompt.jinja"

    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["template_shas"]["prompt.jinja"] == digest
