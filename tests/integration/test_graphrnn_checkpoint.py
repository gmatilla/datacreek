import hashlib
from pathlib import Path

import boto3
import pytest

from datacreek.analysis.fractal import ensure_graphrnn_checkpoint


class DummyClient:
    def __init__(self, data: bytes):
        self.data = data
        self.calls = []

    def download_file(self, bucket, key, dest, ExtraArgs=None):
        self.calls.append((bucket, key, dest))
        Path(dest).write_bytes(self.data)


def test_checkpoint_download(monkeypatch, tmp_path):
    data = b"model"
    sha = hashlib.sha256(data).hexdigest()
    client = DummyClient(data)
    monkeypatch.setattr(boto3, "client", lambda *a, **k: client)
    cfg = {
        "tpl": {
            "rnn_ckpt_bucket": "b",
            "rnn_ckpt_key": "m.pt",
            "rnn_ckpt_sha": sha,
        }
    }
    path = ensure_graphrnn_checkpoint(cfg, cache_dir=tmp_path)
    assert path is not None
    assert Path(path).read_bytes() == data
    assert client.calls
