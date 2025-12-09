import sys
import types

import numpy as np

from datacreek.core.knowledge_graph import KnowledgeGraph


def make_graph():
    kg = KnowledgeGraph()
    kg.add_document("d1", "src")
    kg.add_chunk("d1", "c1", "a")
    kg.add_chunk("d1", "c2", "b")
    kg.graph.add_edge("c1", "c2")
    kg.graph.nodes["c1"]["embedding"] = [1.0, 0.0]
    kg.graph.nodes["c2"]["embedding"] = [0.0, 1.0]
    return kg


def patch_analysis(monkeypatch):
    fake_fractal = types.SimpleNamespace(
        spectral_gap=lambda g, normed=True: 1.0,
        laplacian_energy=lambda g, normed=True: 2.0,
        graph_lacunarity=lambda g, radius=1: 3.0,
        box_counting_dimension=lambda g, radii: (0.9, [(1, 2)]),
        laplacian_spectrum=lambda g, normed=True: np.array([0, 1]),
        spectral_density=lambda g, bins=50, normed=True: (
            np.array([0, 1]),
            np.array([1, 0]),
        ),
        graph_fourier_transform=lambda g, s, normed=True: np.array([1, 2]),
        inverse_graph_fourier_transform=lambda g, c, normed=True: np.array([1, 2]),
        persistence_entropy=lambda g, dimension=0: 4.0,
        persistence_diagrams=lambda g, max_dim=2: {0: np.array([[0, 1]])},
        persistence_wasserstein_distance=lambda g1, g2, dimension=0, order=1: 0.5,
        mdl_optimal_radius=lambda counts: 0,
        fractal_information_metrics=lambda g, radii, max_dim=1: {"metric": 1},
        fractal_information_density=lambda g, radii, max_dim=1: 0.9,
        fractal_level_coverage=lambda g: 0.5,
    )
    fake_sheaf = types.SimpleNamespace(
        sheaf_laplacian=lambda g, edge_attr="sheaf_sign": np.eye(len(g)),
        sheaf_convolution=lambda g, feats, edge_attr="sheaf_sign", alpha=0.1: {
            k: np.asarray(v) for k, v in feats.items()
        },
        sheaf_neural_network=lambda g, feats, layers=2, alpha=0.1, edge_attr="sheaf_sign": {
            k: np.asarray(v) for k, v in feats.items()
        },
        sheaf_first_cohomology=lambda g, edge_attr="sheaf_sign", tol=1e-5: 2,
        sheaf_first_cohomology_blocksmith=lambda g, edge_attr="sheaf_sign", block_size=40000, tol=1e-5: 3,
        resolve_sheaf_obstruction=lambda g, edge_attr="sheaf_sign", max_iter=10: 1,
        sheaf_consistency_score=lambda g, edge_attr="sheaf_sign": 0.9,
        sheaf_consistency_score_batched=lambda g, batches, edge_attr="sheaf_sign": [
            0.1 for _ in batches
        ],
        spectral_bound_exceeded=lambda g, k, tau, edge_attr="sheaf_sign": True,
    )
    monkeypatch.setitem(sys.modules, "datacreek.analysis.fractal", fake_fractal)
    monkeypatch.setitem(sys.modules, "datacreek.analysis.sheaf", fake_sheaf)
    import networkx as nx

    monkeypatch.setattr(nx, "number_connected_components", lambda g: 1)


def test_analysis_helpers(monkeypatch):
    kg = make_graph()
    patch_analysis(monkeypatch)

    assert kg.spectral_gap() == 1.0
    assert kg.laplacian_energy() == 2.0
    assert kg.lacunarity() == 3.0
    assert kg.sheaf_cohomology() == 2
    assert kg.sheaf_cohomology_blocksmith() == 3
    assert kg.resolve_sheaf_obstruction() == 1
    assert kg.sheaf_consistency_score() == 0.9
    assert kg.sheaf_consistency_score_batched([["c1", "c2"]]) == [0.1]
    assert kg.spectral_bound_exceeded(1, 0.5)
    assert np.array_equal(kg.laplacian_spectrum(), np.array([0, 1]))
    d, h = kg.spectral_density()
    assert d.tolist() == [0, 1] and h.tolist() == [1, 0]
    coeffs = kg.graph_fourier_transform({"c1": 1.0, "c2": 0.0})
    assert np.array_equal(kg.inverse_graph_fourier_transform(coeffs), coeffs)
    assert kg.persistence_entropy() == 4.0
    assert np.array_equal(kg.persistence_diagrams()[0], np.array([[0, 1]]))
    assert kg.persistence_wasserstein_distance(kg.graph) == 0.5
    sig = kg.topological_signature(1)
    assert "diagrams" in sig and "entropy" in sig
    assert len(kg.topological_signature_hash(1)) == 32
    assert kg.betti_number(0) == 1
    feats = kg.compute_fractal_features([1])
    assert feats["dimension"] == 0.9 or feats["dimension"] == feats["dimension"]
    assert kg.fractal_information_metrics([1]) == {"metric": 1}
    assert kg.fractal_information_density([1]) == 0.9
    assert kg.ensure_fractal_coverage(0.4, [1]) >= 0.5
