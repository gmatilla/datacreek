import importlib

import pytest

import datacreek.core as core


def test_core_getattr():
    assert core.AppContext.__name__ == "AppContext"
    with pytest.raises(AttributeError):
        core.__getattr__("missing")
