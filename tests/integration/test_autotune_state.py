import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datacreek import AutoTuneState
from datacreek.analysis.autotune import autotune_step, recall_at_k


def test_autotune_state_defaults():
    state = AutoTuneState()
    assert hasattr(state, "alpha")
    assert hasattr(state, "gamma")
    assert hasattr(state, "eta")


def test_recall_and_autotune_step():
    g = nx.Graph()
    g.add_node(
        "a",
        embedding=[1.0, 0.0],
        graphwave_embedding=[1.0, 0.0],
        poincare_embedding=[0.0, 0.5],
    )
    g.add_node(
        "b",
        embedding=[0.0, 1.0],
        graphwave_embedding=[0.0, 1.0],
        poincare_embedding=[0.0, 0.6],
    )
    g.add_edge("a", "b")

    rec = recall_at_k(g, ["a"], {"a": ["b"]}, k=1)
    assert rec == 1.0

    state = AutoTuneState()
    emb = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    labels = {"a": 0, "b": 1}
    res = autotune_step(
        g,
        emb,
        labels,
        [],
        state,
        recall_data=(["a"], {"a": ["b"]}),
        k=1,
        penalty_cfg={
            "lambda_sigma": 1.0,
            "lambda_cov": 1.0,
            "w_rec": 1.0,
            "w_lat": 1.0,
        },
        latency=0.2,
    )
    assert res["recall"] == 1.0


def test_update_theta_sets_gauge(monkeypatch):
    state = AutoTuneState()
    metrics = {
        "cost": 0.5,
        "tau": 3,
        "eps": 0.1,
        "beta": 0.2,
        "delta": 0.05,
        "jitter": 0.1,
    }

    from datacreek.analysis import monitoring
    from datacreek.analysis.autotune import update_theta

    called = {}
    monkeypatch.setattr(
        monitoring,
        "j_cost",
        type("G", (), {"set": lambda self, v: called.setdefault("v", v)})(),
    )
    update_theta(state, metrics)
    assert called["v"] == 0.5
    assert state.tau == 3


def test_jitter_restart_counter(monkeypatch):
    state = AutoTuneState()
    state.stagnation = 4
    state.prev_costs = [0.0]

    g = nx.Graph()
    g.add_edge("a", "b")
    emb = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    labels = {"a": 0, "b": 1}

    import datacreek.analysis.autotune as auto

    monkeypatch.setattr(auto, "structural_entropy", lambda *a, **k: 0.0)
    monkeypatch.setattr(auto, "fractal_level_coverage", lambda *a, **k: 0.0)
    monkeypatch.setattr(auto, "bottleneck_distance", lambda *a, **k: 0.0)
    monkeypatch.setattr(auto, "graph_information_bottleneck", lambda *a, **k: 0.0)
    monkeypatch.setattr(auto, "mdl_description_length", lambda *a, **k: 0.0)
    monkeypatch.setattr(auto, "hybrid_score", lambda *a, **k: 0.0)

    called = {"n": 0}
    monkeypatch.setattr(
        auto.monitoring,
        "gp_jitter_restarts_total",
        type("C", (), {"inc": lambda self: called.__setitem__("n", called["n"] + 1)})(),
    )

    res = autotune_step(g, emb, labels, [g], state)
    assert res["restart_gp"]
    assert called["n"] == 1
