import importlib
import sys

import pytest

pytest.importorskip("PIL")
from PIL import Image

# ensure real libs in case other tests stubbed them
sys.modules.pop("imagehash", None)
imagehash = importlib.import_module("imagehash")

dedup = importlib.import_module("datacreek.utils.image_dedup")


def test_hashes(tmp_path):
    data = b"abc"
    hs = list(dedup._hashes(data))
    assert len(hs) == dedup.K
    assert all(0 <= h < dedup.M for h in hs)


def test_check_duplicate(tmp_path, monkeypatch):
    monkeypatch.setattr(dedup, "FILTER", bytearray(dedup._BYTES))
    img1 = tmp_path / "a.png"
    img2 = tmp_path / "b.png"
    Image.new("RGB", (8, 8), color="red").save(img1)
    Image.new("RGB", (8, 8), color="red").save(img2)
    assert not dedup.check_duplicate(str(img1))
    assert dedup.check_duplicate(str(img2))
