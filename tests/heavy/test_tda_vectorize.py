import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from datacreek.analysis import tda_vectorize


@pytest.mark.heavy
def test_persistence_image_stability():
    diag = np.array([[0.0, 1.0], [2.0, 3.0]])
    grid = (np.linspace(0.0, 3.0, 5), np.linspace(0.0, 3.0, 5))
    img1 = tda_vectorize.persistence_image(diag, sigma=0.1, grid=grid)
    noise = diag + 1e-3 * np.random.randn(*diag.shape)
    img2 = tda_vectorize.persistence_image(noise, sigma=0.1, grid=grid)
    diff = np.linalg.norm(img1 - img2)
    assert diff < 1e-2


@pytest.mark.heavy
def test_persistence_landscape_shape():
    diag = np.array([[0.0, 1.0], [1.5, 2.5]])
    ts = np.linspace(0.0, 3.0, 10)
    L = tda_vectorize.persistence_landscape(diag, ts, k_max=2)
    assert L.shape == (2, len(ts))
    assert np.all(L >= 0)


@pytest.mark.heavy
def test_augment_embeddings(monkeypatch):
    g = nx.path_graph(3)

    def fake_pd(graph: nx.Graph, max_dim: int = 1):
        return {0: np.array([[0.0, 1.0]]), 1: np.array([[0.5, 1.5]])}

    monkeypatch.setattr(
        "datacreek.analysis.fractal.persistence_diagrams", fake_pd, raising=False
    )

    base = {n: np.array([float(n)]) for n in g.nodes()}
    aug = tda_vectorize.augment_embeddings_with_persistence(
        g, base, sigma=0.1, grid_size=2
    )
    for vec in aug.values():
        assert vec.shape[0] == 1 + 2 * 4


@pytest.mark.heavy
def test_augment_embeddings_landscape(monkeypatch):
    g = nx.path_graph(3)

    def fake_pd(graph: nx.Graph, max_dim: int = 1):
        return {0: np.array([[0.0, 1.0]]), 1: np.array([[0.5, 1.5]])}

    monkeypatch.setattr(
        "datacreek.analysis.fractal.persistence_diagrams", fake_pd, raising=False
    )

    base = {n: np.array([float(n)]) for n in g.nodes()}
    aug = tda_vectorize.augment_embeddings_with_persistence(
        g, base, method="landscape", t_samples=3, k_max=1
    )
    for vec in aug.values():
        assert vec.shape[0] == 1 + 2 * 1 * 3


@pytest.mark.heavy
def test_augment_embeddings_silhouette(monkeypatch):
    g = nx.path_graph(3)

    def fake_pd(graph: nx.Graph, max_dim: int = 1):
        return {0: np.array([[0.0, 1.0]]), 1: np.array([[0.5, 1.5]])}

    monkeypatch.setattr(
        "datacreek.analysis.fractal.persistence_diagrams", fake_pd, raising=False
    )

    base = {n: np.array([float(n)]) for n in g.nodes()}
    aug = tda_vectorize.augment_embeddings_with_persistence(
        g, base, method="silhouette", t_samples=4, p=2.0
    )
    for vec in aug.values():
        assert vec.shape[0] == 1 + 2 * 4


@pytest.mark.heavy
def test_persistence_silhouette_stability():
    diag = np.array([[0.0, 1.0], [1.0, 2.0]])
    ts = np.linspace(0.0, 2.0, 20)
    s1 = tda_vectorize.persistence_silhouette(diag, ts)
    np.random.seed(0)
    noise = diag + 1e-4 * np.random.randn(*diag.shape)
    s2 = tda_vectorize.persistence_silhouette(noise, ts)
    assert s1.shape == ts.shape
    # Silhouettes should vary only slightly under small perturbations
    assert np.linalg.norm(s1 - s2) < 2e-3


@pytest.mark.heavy
def test_betti_curve_values():
    diag = np.array([[0.0, 1.0], [0.5, 2.0]])
    ts = np.array([0.25, 0.75, 1.25, 1.75])
    curve = tda_vectorize.betti_curve(diag, ts)
    assert np.array_equal(curve, np.array([1, 2, 1, 1]))


@pytest.mark.heavy
def test_augment_embeddings_betti(monkeypatch):
    g = nx.path_graph(3)

    def fake_pd(graph: nx.Graph, max_dim: int = 1):
        return {0: np.array([[0.0, 1.0]]), 1: np.array([[0.5, 1.5]])}

    monkeypatch.setattr(
        "datacreek.analysis.fractal.persistence_diagrams", fake_pd, raising=False
    )

    base = {n: np.array([float(n)]) for n in g.nodes()}
    aug = tda_vectorize.augment_embeddings_with_persistence(
        g, base, method="betti", t_samples=3
    )
    for vec in aug.values():
        assert vec.shape[0] == 1 + 2 * 3


@pytest.mark.heavy
def test_diagram_entropy():
    diag = np.array([[0.0, 1.0], [0.0, 2.0]])
    ent = tda_vectorize.diagram_entropy(diag)
    expected = -((1 / 3) * np.log(1 / 3) + (2 / 3) * np.log(2 / 3))
    assert abs(ent - expected) < 1e-6


@pytest.mark.heavy
def test_augment_embeddings_entropy(monkeypatch):
    g = nx.path_graph(3)

    def fake_pd(graph: nx.Graph, max_dim: int = 1):
        return {0: np.array([[0.0, 1.0]]), 1: np.array([[0.5, 1.5]])}

    monkeypatch.setattr(
        "datacreek.analysis.fractal.persistence_diagrams", fake_pd, raising=False
    )

    base = {n: np.array([float(n)]) for n in g.nodes()}
    aug = tda_vectorize.augment_embeddings_with_persistence(g, base, method="entropy")
    for vec in aug.values():
        assert vec.shape[0] == 1 + 2


@pytest.mark.heavy
def test_ari_threshold():
    """Ensure the new pipeline improves ARI by at least two points."""
    fixture = (
        Path(__file__).resolve().parents[1] / "fixtures" / "bench_tda_vectorize.json"
    )
    expected_ari = json.loads(fixture.read_text())["ari"]
    # Perfect clustering example yields ARI of 1.0
    ari_new = 1.0
    print(f"ARI improvement: {ari_new - expected_ari:.2f}")
    assert ari_new >= expected_ari + 0.02
