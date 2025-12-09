import math
import sys
import types

import networkx as nx
import numpy as np
import pytest

from datacreek.analysis import generation as gen


@pytest.mark.heavy
def test_generate_graph_rnn_like_edges():
    g = gen.generate_graph_rnn_like(5, 4)
    assert g.number_of_nodes() == 5
    # random graph may have more edges than requested but not fewer
    assert g.number_of_edges() >= 4


@pytest.mark.heavy
def test_generate_graph_rnn(monkeypatch):
    monkeypatch.setattr(gen.random, "random", lambda: 0.0)
    g = gen.generate_graph_rnn(4, 3, p=1.0)
    assert g.number_of_nodes() == 4
    assert g.number_of_edges() == 3


@pytest.mark.heavy
def test_generate_graph_rnn_stateful():
    g = gen.generate_graph_rnn_stateful(4, 3, seed=0)
    assert g.number_of_nodes() == 4
    assert g.number_of_edges() <= 3


@pytest.mark.heavy
def test_generate_netgan_like():
    base = nx.path_graph(4)
    g = gen.generate_netgan_like(base, num_walks=2, walk_length=3, p=1.0)
    assert g.number_of_nodes() == 4
    assert g.number_of_edges() > 0


@pytest.mark.heavy
def test_bias_reweighting_basic(monkeypatch):
    """bias_reweighting should upweight under-represented keys."""
    neigh = {"A": 1, "B": 2}
    glob = {"A": 2, "B": 2}
    weights = {"A": 0.5, "B": 0.5}

    scipy_stub = types.SimpleNamespace(
        stats=types.SimpleNamespace(wasserstein_distance=lambda a, b: 1.0)
    )
    monkeypatch.setitem(sys.modules, "scipy", scipy_stub)
    monkeypatch.setitem(sys.modules, "scipy.stats", scipy_stub.stats)

    out = gen.bias_reweighting(neigh, glob, weights, threshold=0.0)
    assert out["A"] > weights["A"]


@pytest.mark.heavy
def test_sheaf_score_and_apply(monkeypatch):
    """sheaf_score and apply_logit_bias should run without heavy deps."""
    scipy_stub = types.SimpleNamespace(
        sparse=types.SimpleNamespace(
            csr_matrix=lambda x: np.asarray(x),
            linalg=types.SimpleNamespace(
                cg=lambda A, b, atol=1e-6, rtol=1e-5, maxiter=1000: (np.array(b), 0)
            ),
        )
    )
    monkeypatch.setitem(sys.modules, "scipy", scipy_stub)
    monkeypatch.setitem(sys.modules, "scipy.sparse", scipy_stub.sparse)
    monkeypatch.setitem(sys.modules, "scipy.sparse.linalg", scipy_stub.sparse.linalg)

    Delta = np.eye(2)
    b = [1.0, 0.0]
    score = gen.sheaf_score(b, Delta)
    assert score == pytest.approx(1.0)

    monkeypatch.setattr(gen, "bias_wasserstein", lambda a, b, c: (c, 0.2))
    payload = {"logits": [0.1, 0.9]}
    W = gen.apply_logit_bias(payload, [1], [1])
    assert W == 0.2
    assert payload["logits"] == [0.1, 0.9]


@pytest.mark.heavy
def test_generate_graph_rnn_sequential():
    g = gen.generate_graph_rnn_sequential(4, 2, seed=0, directed=False)
    assert not g.is_directed()
    assert g.number_of_nodes() == 4
    assert g.number_of_edges() <= 2


@pytest.mark.heavy
def test_sheaf_consistency_and_bias(monkeypatch):
    """sheaf_consistency_real and bias_wasserstein should handle stubs."""
    g = nx.path_graph(2)
    monkeypatch.setattr(
        "datacreek.analysis.sheaf.sheaf_laplacian",
        lambda graph, edge_attr="sheaf_sign": np.eye(len(graph)),
    )

    scipy_stub = types.SimpleNamespace(
        sparse=types.SimpleNamespace(
            csr_matrix=lambda x: np.asarray(x),
            linalg=types.SimpleNamespace(
                cg=lambda A, b, atol=1e-6, rtol=1e-5, maxiter=1000: (np.array(b), 0)
            ),
        )
    )
    monkeypatch.setitem(sys.modules, "scipy", scipy_stub)
    monkeypatch.setitem(sys.modules, "scipy.sparse", scipy_stub.sparse)
    monkeypatch.setitem(sys.modules, "scipy.sparse.linalg", scipy_stub.sparse.linalg)

    score = gen.sheaf_consistency_real(g, [1, 0])
    assert score == pytest.approx(1.0)

    monkeypatch.setitem(
        sys.modules,
        "geomloss",
        types.SimpleNamespace(SamplesLoss=lambda *a, **k: (lambda x, y: 0.2)),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            as_tensor=lambda x, dtype=None: np.asarray(x, dtype=float),
            float32=np.float32,
        ),
    )
    scaled, W = gen.bias_wasserstein([1], [1], [1.0])
    assert W == 0.2
    assert scaled[0] == pytest.approx(math.exp(-0.2))


@pytest.mark.heavy
def test_format_helpers(monkeypatch):
    monkeypatch.setattr(
        gen, "bias_wasserstein", lambda a, b, c: (np.array(c) * 0.5, 0.1)
    )
    payload = {"logits": [1.0, 2.0]}
    chat = gen.generate_chatml(dict(payload), [], [])
    alpaca = gen.generate_alpaca(dict(payload), [], [])
    assert chat["logits"] == [0.5, 1.0]
    assert alpaca["logits"] == [0.5, 1.0]
