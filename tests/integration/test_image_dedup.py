import importlib
import sys

import pytest

pytest.importorskip("PIL")
from PIL import Image

# Ensure real scipy is available because other tests may stub it
sys.modules.pop("scipy", None)
sys.modules.pop("scipy.fftpack", None)
importlib.import_module("scipy.fftpack")

from datacreek.utils.image_dedup import FILTER, check_duplicate


def test_check_duplicate(tmp_path):
    img1 = tmp_path / "a.png"
    img2 = tmp_path / "b.png"
    Image.new("RGB", (8, 8), color="red").save(img1)
    Image.new("RGB", (8, 8), color="red").save(img2)

    FILTER[:] = b"\x00" * len(FILTER)
    assert not check_duplicate(str(img1))
    assert check_duplicate(str(img2))
