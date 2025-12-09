import logging
import random
import sys
import types

import networkx as nx
import numpy as np
import pytest

from datacreek.analysis import generation


def test_generate_graph_rnn_like_deterministic():
    g1 = generation.generate_graph_rnn_like(5, 4)
    g2 = nx.gnm_random_graph(5, 4, seed=0)
    assert sorted(g1.edges()) == sorted(g2.edges())


def test_generate_graph_rnn_all_edges(monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.0)
    g = generation.generate_graph_rnn(4, 3, p=0.5)
    assert g.number_of_edges() == 3
    # edges are undirected so orientations may vary
    valid = {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
    assert set(map(tuple, map(sorted, g.edges()))) <= set(
        map(tuple, map(sorted, valid))
    )


def test_generate_graph_rnn_stateful_seed():
    g = generation.generate_graph_rnn_stateful(4, 2, hidden_dim=2, seed=0)
    assert g.number_of_edges() <= 2
    assert list(g.nodes()) == [0, 1, 2, 3]


def test_generate_graph_rnn_sequential_directed():
    g = generation.generate_graph_rnn_sequential(4, 3, hidden_dim=2, seed=1)
    assert g.number_of_edges() <= 3
    for u, v in g.edges():
        assert u > v  # edges from newer to older


def test_generate_netgan_like_empty_graph():
    empty = nx.Graph()
    new_g = generation.generate_netgan_like(empty)
    assert new_g.number_of_edges() == 0


def test_generate_netgan_like_walks():
    g = nx.cycle_graph(3)
    new_g = generation.generate_netgan_like(g, num_walks=5, walk_length=4, p=1.0)
    assert new_g.number_of_edges() > 0


def test_bias_reweighting_upweights(monkeypatch):
    dummy_stats = types.SimpleNamespace(wasserstein_distance=lambda a, b: 0.5)
    monkeypatch.setitem(sys.modules, "scipy.stats", dummy_stats)
    neighbors = {"A": 1}
    global_demog = {"A": 1, "B": 1}
    weights = {"A": 1.0, "B": 1.0}
    adjusted = generation.bias_reweighting(
        neighbors, global_demog, weights, threshold=0.1
    )
    assert pytest.approx(adjusted["B"], abs=1e-6) == 1.2


def test_sheaf_consistency_real(monkeypatch):
    def fake_laplacian(g, edge_attr="sheaf_sign"):
        return np.eye(2)

    dummy_cg = lambda A, b, atol=1e-6: (np.zeros_like(b), None)
    monkeypatch.setitem(
        sys.modules, "scipy.sparse", types.SimpleNamespace(csr_matrix=lambda x: x)
    )
    monkeypatch.setitem(
        sys.modules, "scipy.sparse.linalg", types.SimpleNamespace(cg=dummy_cg)
    )
    monkeypatch.setitem(
        sys.modules,
        "datacreek.analysis.sheaf",
        types.SimpleNamespace(sheaf_laplacian=fake_laplacian),
    )
    g = nx.Graph()
    g.add_nodes_from([0, 1])
    g.add_edge(0, 1, sheaf_sign=1)
    score = generation.sheaf_consistency_real(g, [1.0, 1.0])
    assert pytest.approx(score, abs=1e-6) == 1.0 / (1.0 + np.sqrt(2))


def test_apply_logit_bias(monkeypatch, caplog):
    def fake_bw(loc, glob, logits):
        return [l * 0.5 for l in logits], 0.2

    monkeypatch.setattr(generation, "bias_wasserstein", fake_bw)
    caplog.set_level(logging.INFO)
    payload = {"logits": [2.0, 4.0]}
    W = generation.apply_logit_bias(payload, [1, 1], [1, 1])
    assert W == 0.2
    assert payload["logits"] == [1.0, 2.0]
    assert any("Bias factor" in rec.message for rec in caplog.records)


def test_bias_wasserstein_scaling(monkeypatch):
    torch_stub = types.SimpleNamespace(
        as_tensor=lambda x, dtype=None: np.asarray(x, dtype=float), float32=None
    )
    geom_stub = types.SimpleNamespace(
        SamplesLoss=lambda *_a, **_k: (lambda a, b: np.array(0.3))
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "geomloss", geom_stub)
    scaled, w = generation.bias_wasserstein([0, 1], [1, 0], [2.0, 2.0])
    assert pytest.approx(w, abs=1e-6) == 0.3
    assert all(pytest.approx(v) == v for v in scaled)


def test_generate_chatml_alpaca(monkeypatch):
    monkeypatch.setattr(
        generation, "bias_wasserstein", lambda a, b, c: ([x * 0.5 for x in c], 0.1)
    )
    p = {"logits": [4, 4]}
    assert generation.generate_chatml(p.copy(), [1], [1])["logits"] == [2, 2]
    assert generation.generate_alpaca(p.copy(), [1], [1])["logits"] == [2, 2]


def test_sheaf_score(monkeypatch):
    dummy_cg = lambda A, b, **k: (np.zeros_like(b), None)
    monkeypatch.setitem(
        sys.modules, "scipy.sparse.linalg", types.SimpleNamespace(cg=dummy_cg)
    )
    Delta = np.eye(2)
    score = generation.sheaf_score([1.0, 1.0], Delta)
    expected = 1.0 / (1.0 + np.linalg.norm([1.0, 1.0]))
    assert pytest.approx(score, abs=1e-6) == expected
