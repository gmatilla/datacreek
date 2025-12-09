import importlib
import sys
import types
from contextlib import contextmanager

import pytest


def reload_utils(monkeypatch, rich_module=None):
    monkeypatch.syspath_prepend("datacreek")
    if rich_module is not None:
        monkeypatch.setitem(sys.modules, "rich.progress", rich_module)
    elif "rich.progress" in sys.modules:
        monkeypatch.delitem(sys.modules, "rich.progress")
    if "datacreek.utils" in sys.modules:
        del sys.modules["datacreek.utils"]
    import datacreek.utils as utils

    importlib.reload(utils)
    return utils


def test_progress_no_rich(monkeypatch):
    utils = reload_utils(monkeypatch)
    with utils.progress_context("a", 1) as ctx:
        assert ctx == (None, 0)


def test_progress_with_stub(monkeypatch):
    class FakeProgress:
        __module__ = "rich.progress"

    rich_mod = types.SimpleNamespace(Progress=FakeProgress)
    # stub progress module with simple functions
    prog = types.SimpleNamespace(
        create_progress=lambda *a, **k: ("p", 1),
        progress_context=contextmanager(lambda *a, **k: iter([("p", 1)])),
    )
    monkeypatch.setitem(sys.modules, "datacreek.utils.progress", prog)
    utils = reload_utils(monkeypatch, rich_mod)
    assert utils.create_progress("a", 1) == ("p", 1)
    with utils.progress_context("b", 2) as ctx:
        assert ctx == ("p", 1)


def test_extract_entities(monkeypatch):
    fake = types.SimpleNamespace(extract_entities=lambda x: ["e"])
    monkeypatch.setitem(sys.modules, "datacreek.utils.entity_extraction", fake)
    utils = reload_utils(monkeypatch)
    assert utils.__getattr__("extract_entities")("t") == ["e"]
