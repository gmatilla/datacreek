import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import networkx as nx
import pytest

from datacreek.analysis.fractal import (
    bottleneck_distance,
    box_counting_dimension,
    box_cover,
    fractalize_graph,
    fractalize_optimal,
    graph_fourier_transform,
    graphwave_embedding,
    graphwave_embedding_chebyshev,
    graphwave_entropy,
    inverse_graph_fourier_transform,
    mdl_optimal_radius,
    persistence_diagrams,
    persistence_entropy,
    persistence_wasserstein_distance,
    poincare_embedding,
    spectral_density,
    spectral_dimension,
)


def test_box_counting_dimension_grid():
    g = nx.grid_2d_graph(4, 4)
    mapping = {node: i for i, node in enumerate(g.nodes())}
    g = nx.relabel_nodes(g, mapping)
    dim, counts = box_counting_dimension(g, [1, 2, 3])
    assert dim > 0
    assert counts


def test_box_cover_matches_counts():
    g = nx.path_graph(5)
    boxes = box_cover(g, 1)
    nodes = sorted(n for box in boxes for n in box)
    assert nodes == list(range(g.number_of_nodes()))
    _, counts = box_counting_dimension(g, [1])
    assert counts[0] == (1, len(boxes))


def test_fractalize_graph_simple():
    g = nx.path_graph(4)
    coarse, mapping = fractalize_graph(g, 1)
    assert coarse.number_of_nodes() == 2
    assert coarse.number_of_edges() == 1
    assert len(mapping) == 4


def test_fractalize_optimal():
    g = nx.path_graph(6)
    coarse, mapping, radius = fractalize_optimal(g, [1, 2])
    assert radius in {1, 2}
    assert len(mapping) == g.number_of_nodes()
    assert coarse.number_of_nodes() >= 1


def test_persistence_entropy_path():
    g = nx.path_graph(5)
    ent = persistence_entropy(g, dimension=0)
    assert ent > 0


def test_graphwave_embedding_shape():
    g = nx.path_graph(4)
    emb = graphwave_embedding(g, scales=[0.5], num_points=4)
    assert len(emb) == g.number_of_nodes()
    for vec in emb.values():
        assert vec.shape == (8,)


def test_graphwave_embedding_chebyshev_shape():
    g = nx.path_graph(4)
    try:
        emb = graphwave_embedding_chebyshev(g, scales=[0.5], num_points=4, order=3)
    except ModuleNotFoundError:
        pytest.skip("scipy not installed")
    assert len(emb) == g.number_of_nodes()
    for vec in emb.values():
        assert vec.shape == (8,)


def test_bottleneck_distance_identical():
    g1 = nx.path_graph(4)
    g2 = nx.path_graph(4)
    assert bottleneck_distance(g1, g2) == pytest.approx(0.0, abs=1e-9)


def test_bottleneck_distance_different():
    g1 = nx.path_graph(4)
    g2 = nx.cycle_graph(4)
    assert bottleneck_distance(g1, g2) > 0.0


def test_mdl_optimal_radius():
    g = nx.path_graph(6)
    _, counts = box_counting_dimension(g, [1, 2, 3])
    idx = mdl_optimal_radius(counts)
    assert 0 <= idx < len(counts)


def test_poincare_embedding_shape():
    g = nx.DiGraph([(0, 1), (1, 2), (1, 3)])
    emb = poincare_embedding(g, dim=2, negative=2, epochs=5)
    assert len(emb) == g.number_of_nodes()
    for vec in emb.values():
        assert vec.shape == (2,)


def test_spectral_dimension_grid():
    g = nx.grid_2d_graph(3, 3)
    mapping = {node: i for i, node in enumerate(g.nodes())}
    g = nx.relabel_nodes(g, mapping)
    dim, traces = spectral_dimension(g, [0.1, 0.2, 0.4])
    assert dim > 0
    assert traces


def test_persistence_diagrams():
    g = nx.path_graph(4)
    diags = persistence_diagrams(g, max_dim=1)
    assert 0 in diags and 1 in diags
    for diag in diags.values():
        assert diag.shape[1] == 2


def test_persistence_diagrams_triangle_clique():
    g = nx.complete_graph(3)
    diags = persistence_diagrams(g, max_dim=1)
    # the 1-simplex is filled, so no 1-dimensional holes
    assert diags[1].size == 0


def test_spectral_density():
    g = nx.path_graph(4)
    hist, edges = spectral_density(g, bins=4)
    assert len(hist) == 4
    assert len(edges) == 5
    assert hist.sum() > 0


def test_graph_fourier_roundtrip():
    g = nx.path_graph(4)
    signal = {n: float(n) for n in g.nodes()}
    coeffs = graph_fourier_transform(g, signal)
    recon = inverse_graph_fourier_transform(g, coeffs)
    for n, val in zip(g.nodes(), recon):
        assert pytest.approx(val, rel=1e-6) == signal[n]


def test_persistence_wasserstein_distance():
    g1 = nx.path_graph(4)
    g2 = nx.cycle_graph(4)
    if (
        persistence_wasserstein_distance.__module__ == "datacreek.analysis.fractal"
        and getattr(__import__("datacreek.analysis.fractal", fromlist=["gd"]), "gd")
        is None
    ):
        pytest.skip("gudhi not available")
    d = persistence_wasserstein_distance(g1, g1)
    assert d == pytest.approx(0.0, abs=1e-9)
    assert persistence_wasserstein_distance(g1, g2) > 0.0


def test_graphwave_entropy_formula():
    emb = {0: [3.0, 4.0], 1: [0.0, 1.0]}
    val = graphwave_entropy(emb)
    expected = -0.5 * (math.log(5.0) + math.log(1.0))
    assert val == pytest.approx(expected)
