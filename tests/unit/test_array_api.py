import importlib
import types

import pytest

import datacreek.backend.array_api as array_api


class DummyArray:
    __class__ = type("Dummy", (), {"__module__": "cupy"})


def test_get_xp_from_obj(monkeypatch):
    np_mod = types.SimpleNamespace(name="numpy")
    cp_mod = types.SimpleNamespace(name="cupy")

    def fake_import(name):
        return cp_mod if name == "cupy" else np_mod

    monkeypatch.setattr(array_api, "import_module", fake_import)
    assert array_api.get_xp(DummyArray()) is cp_mod


def test_get_xp_cupy_available(monkeypatch):
    np_mod = types.SimpleNamespace(name="numpy")
    cp_mod = types.SimpleNamespace(name="cupy")
    monkeypatch.setattr(
        array_api, "import_module", lambda n: cp_mod if n == "cupy" else np_mod
    )
    monkeypatch.setattr(array_api, "_cupy_available", lambda: True)
    assert array_api.get_xp() is cp_mod


def test_get_xp_fallback(monkeypatch):
    np_mod = types.SimpleNamespace(name="numpy")
    monkeypatch.setattr(array_api, "_cupy_available", lambda: False)
    monkeypatch.setattr(array_api, "import_module", lambda n: np_mod)
    assert array_api.get_xp() is np_mod


def test_cupy_available_success(monkeypatch):
    module = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            runtime=types.SimpleNamespace(getDeviceCount=lambda: 1)
        )
    )
    monkeypatch.setattr(array_api, "import_module", lambda n: module)
    assert array_api._cupy_available()


def test_cupy_available_failure(monkeypatch):
    def importer(name):
        raise Exception("no cupy")

    monkeypatch.setattr(array_api, "import_module", importer)
    assert not array_api._cupy_available()
