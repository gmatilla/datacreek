import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np


def test_cache_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DATACREEK_CACHE", str(tmp_path))
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    skl = types.ModuleType("sklearn.cross_decomposition")
    skl.CCA = lambda n_components=2: type(
        "DummyCCA",
        (),
        {
            "x_weights_": np.eye(n_components),
            "y_weights_": np.eye(n_components),
        },
    )()
    sys.modules["sklearn.cross_decomposition"] = skl
    sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
    spec = importlib.util.spec_from_file_location(
        "mv", root / "datacreek" / "analysis" / "multiview.py"
    )
    mv = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mv)
    assert mv.CACHE_ROOT == str(tmp_path)
    assert os.path.join(str(tmp_path), "cca.pkl") == mv.load_cca.__defaults__[0]
