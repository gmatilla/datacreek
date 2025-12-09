import importlib
import sys
import types

import pytest

if isinstance(sys.modules.get("PIL"), types.SimpleNamespace):
    pytest.skip("pillow unavailable", allow_module_level=True)

from PIL import Image

from datacreek.utils.image_quality import should_caption


def test_should_caption(tmp_path):
    flat = tmp_path / "f.png"
    Image.new("L", (8, 8), color=128).save(flat)

    grid = tmp_path / "g.png"
    img = Image.new("L", (8, 8))
    for i in range(8):
        for j in range(8):
            img.putpixel((i, j), 0 if (i + j) % 2 == 0 else 255)
    img.save(grid)

    assert should_caption(str(flat), threshold=0.4) is False
    assert should_caption(str(grid), threshold=0.4) is True
