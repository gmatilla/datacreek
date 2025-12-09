import importlib
import sys
import types

import datacreek.utils.text as text


def test_normalize_units_no_deps(monkeypatch):
    monkeypatch.setitem(sys.modules, "quantulum3", None)
    monkeypatch.setitem(sys.modules, "pint", None)
    importlib.reload(text)
    assert text.normalize_units("5 km") == "5 km"


def test_normalize_units_stub(monkeypatch):
    class Qty:
        value = 5
        unit = types.SimpleNamespace(name="kilometer")
        span = (0, 4)

    qmod = types.SimpleNamespace(parser=types.SimpleNamespace(parse=lambda t: [Qty()]))
    monkeypatch.setitem(sys.modules, "quantulum3", qmod)

    class DummyConv:
        def __init__(self, mag, units):
            self.magnitude = mag
            self.units = units

        def to_base_units(self):
            return DummyConv(5000, "meter")

    class DummyUnit:
        def __rmul__(self, other):
            return DummyConv(other * 1000, "meter")

    class UReg:
        def __call__(self, name):
            return DummyUnit()

    pint_mod = types.SimpleNamespace(UnitRegistry=lambda: UReg())
    monkeypatch.setitem(sys.modules, "pint", pint_mod)
    importlib.reload(text)

    assert text.normalize_units("5 km") == "5000 meter"
