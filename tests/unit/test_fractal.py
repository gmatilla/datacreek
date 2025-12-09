import time

import networkx as nx
import numpy as np
import pytest

from datacreek.analysis import fractal


class DummyCounter:
    def __init__(self):
        self.count = 0

    def inc(self):
        self.count += 1


class DummyGauge:
    def __init__(self):
        self.value = None

    def set(self, val):
        self.value = val


def test_with_timeout_success():
    counter = DummyCounter()
    gauge = DummyGauge()

    def func():
        return 5.0

    wrapped = fractal.with_timeout(0.1, counter=counter, duration_gauge=gauge)(func)
    assert wrapped() == 5.0
    assert counter.count == 0
    assert gauge.value >= 0


def test_with_timeout_fallback():
    counter = DummyCounter()
    gauge = DummyGauge()

    def slow():
        time.sleep(0.05)
        return 1.0

    wrapped = fractal.with_timeout(
        0.01, counter=counter, duration_gauge=gauge, fallback=lambda: 3.0
    )(slow)
    assert wrapped() == 3.0
    assert counter.count == 1
    assert pytest.approx(gauge.value, rel=0.2) == 0.01


def test_box_cover_and_dimensions():
    G = nx.path_graph(4)
    boxes = fractal.box_cover(G, radius=1)
    # all nodes must be covered
    assert set().union(*boxes) == set(G.nodes())

    def simple_polyfit(x, y, deg):
        a = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(a, y, rcond=None)[0]
        return np.array([m, c])

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(fractal.np, "polyfit", simple_polyfit)

    dim, counts = fractal.box_counting_dimension(G, radii=[1, 2])
    # two radii -> two count entries
    assert len(counts) == 2
    assert dim >= 0

    cdim, ccounts = fractal.colour_box_dimension(G, radii=[1, 2])
    assert len(ccounts) == 2
    assert cdim >= 0

    monkeypatch.undo()


def test_mdl_functions(monkeypatch):
    counts = [(1, 5), (2, 3), (4, 1)]
    monkeypatch.setattr(fractal.np, "polyfit", lambda x, y, d: np.array([0.0, 0.0]))

    assert fractal.mdl_optimal_radius(counts) == 1
    v = fractal.mdl_value(counts)
    s = fractal._slope(counts)
    idx = fractal.dichotomic_radius(counts, target=s)
    assert isinstance(v, float)
    assert 0 <= idx <= len(counts) - 1

    monkeypatch.undo()


def test_graph_lacunarity_and_fourier(monkeypatch):
    """Ensure Fourier transforms round trip correctly and lacunarity > 0."""
    # avoid SciPy sparse path for deterministic tests
    monkeypatch.setattr(fractal, "csgraph", None)
    monkeypatch.setattr(fractal, "eigh", np.linalg.eigh)

    G = nx.cycle_graph(4)
    lac = fractal.graph_lacunarity(G, radius=1)
    assert lac > 0

    signal = {n: float(n) for n in G.nodes()}
    coeffs = fractal.graph_fourier_transform(G, signal)
    recovered = fractal.inverse_graph_fourier_transform(G, coeffs)
    for n, val in signal.items():
        assert pytest.approx(recovered[n]) == val


def test_fractal_information_metrics_and_density(monkeypatch):
    G = nx.path_graph(5)

    def simple_polyfit(x, y, deg):
        a = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(a, y, rcond=None)[0]
        return np.array([m, c])

    monkeypatch.setattr(fractal.np, "polyfit", simple_polyfit)

    monkeypatch.setattr(fractal, "gd", object())
    monkeypatch.setattr(fractal, "gr", object())
    monkeypatch.setattr(fractal, "persistence_entropy", lambda *_a, **_k: 1.0)

    metrics = fractal.fractal_information_metrics(G, radii=[1, 2], max_dim=1)
    assert metrics["dimension"] >= 0
    assert metrics["entropy"] == {0: 1.0, 1: 1.0}

    dens = fractal.fractal_information_density(G, radii=[1, 2], max_dim=1)
    assert dens > 0
