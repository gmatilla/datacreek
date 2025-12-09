import math
import time
import types

import networkx as nx
import numpy as np
import pytest

import datacreek.analysis.fractal as fr


@pytest.mark.heavy
def test_box_cover_and_dimension():
    g = nx.path_graph(4)
    boxes = fr.box_cover(g, radius=1)
    assert all(box for box in boxes)
    dim, counts = fr.box_counting_dimension(g, [1, 2])
    assert len(counts) == 2
    assert dim >= 0


@pytest.mark.heavy
def test_colour_box_and_mdl_functions():
    g = nx.cycle_graph(6)
    dim, counts = fr.colour_box_dimension(g, [1, 2])
    assert dim >= 0
    idx = fr.mdl_optimal_radius(counts)
    assert 0 <= idx < len(counts)
    val = fr.mdl_value(counts)
    assert isinstance(val, float)
    slope = fr._slope(counts)
    assert isinstance(slope, float)
    best = fr.dichotomic_radius(counts, slope)
    assert 0 <= best < len(counts)


@pytest.mark.heavy
def test_fourier_transform_roundtrip():
    g = nx.path_graph(3)
    signal = {n: float(n) for n in g}
    coeffs = fr.graph_fourier_transform(g, signal, normed=False)
    recovered = fr.inverse_graph_fourier_transform(g, coeffs, normed=False)
    vec = np.array(list(signal.values()), dtype=float)
    assert np.allclose(vec, recovered)


@pytest.mark.heavy
def test_graph_lacunarity():
    g = nx.star_graph(4)
    val = fr.graph_lacunarity(g, radius=1)
    assert val > 0


@pytest.mark.heavy
def test_with_timeout_and_density(monkeypatch):
    class Counter:
        def __init__(self):
            self.n = 0

        def inc(self):
            self.n += 1

    class Gauge:
        def __init__(self):
            self.val = None

        def set(self, v):
            self.val = v

    counter = Counter()
    gauge = Gauge()

    wrapped = fr.with_timeout(
        0.01, counter=counter, duration_gauge=gauge, fallback=lambda: 2
    )(lambda: time.sleep(0.02))
    res = wrapped()
    assert res == 2
    assert counter.n == 1
    assert gauge.val == pytest.approx(0.01, rel=0.2)

    g = nx.path_graph(4)
    monkeypatch.setattr(fr, "gd", types.SimpleNamespace())
    monkeypatch.setattr(fr, "gr", types.SimpleNamespace())
    monkeypatch.setattr(fr, "persistence_entropy", lambda *a, **k: 0.0)
    dens = fr.fractal_information_density(g, [1, 2], max_dim=0)
    assert dens >= 0


@pytest.mark.heavy
def test_information_metrics_with_stub(monkeypatch):
    g = nx.path_graph(4)
    monkeypatch.setattr(fr, "gd", types.SimpleNamespace(SimplexTree=lambda: None))
    monkeypatch.setattr(fr, "gr", types.SimpleNamespace())
    monkeypatch.setattr(fr, "persistence_entropy", lambda *a, **k: 0.1)
    metrics = fr.fractal_information_metrics(g, [1, 2], max_dim=0)
    assert metrics["dimension"] >= 0
    assert metrics["entropy"] == {0: 0.1}
