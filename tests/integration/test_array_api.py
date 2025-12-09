import builtins
import importlib
import sys

import numpy as np
import pytest


def test_get_xp_no_cupy(monkeypatch):
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cupy":
            raise ImportError("cupy not installed")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    monkeypatch.delitem(sys.modules, "datacreek.analysis.graphwave_cuda", raising=False)

    from datacreek.backend.array_api import get_xp

    xp = get_xp()
    assert xp is np


def test_import_graphwave_no_cupy(monkeypatch):
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cupy":
            raise ImportError("cupy not installed")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    monkeypatch.delitem(sys.modules, "datacreek.analysis.graphwave_cuda", raising=False)

    module = importlib.import_module("datacreek.analysis.graphwave_cuda")
    assert getattr(module, "cp", None) is None
