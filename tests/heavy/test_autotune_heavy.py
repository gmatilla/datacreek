import types

import networkx as nx
import numpy as np
import pytest

from datacreek.analysis.autotune import (
    AutoTuneState,
    kw_gradient,
    recall_at_k,
    update_theta,
)


@pytest.mark.heavy
def test_recall_at_k_and_update_theta(monkeypatch):
    g = nx.Graph()
    g.add_node(
        "q", embedding=[1, 0], graphwave_embedding=[0, 1], poincare_embedding=[1, 1]
    )
    g.add_node(
        "a", embedding=[1, 0], graphwave_embedding=[0, 1], poincare_embedding=[1, 1]
    )
    g.add_node(
        "b", embedding=[0, 1], graphwave_embedding=[1, 0], poincare_embedding=[1, 1]
    )
    recall = recall_at_k(g, ["q"], {"q": ["a"]}, k=1)
    assert recall == 1.0

    gauge = types.SimpleNamespace(value=None, set=lambda v: setattr(gauge, "value", v))
    monkeypatch.setattr("datacreek.analysis.monitoring.j_cost", gauge, raising=False)
    state = AutoTuneState()
    metrics = {
        "cost": 1.2,
        "tau": 2,
        "eps": 0.1,
        "beta": 0.2,
        "delta": 0.3,
        "jitter": 0.4,
    }
    update_theta(state, metrics)
    assert gauge.value == 1.2
    assert state.tau == 2 and state.eps == 0.1
    assert state.beta == 0.2 and state.delta == 0.3 and state.jitter == 0.4


@pytest.mark.heavy
def test_kw_gradient_deterministic(monkeypatch):
    seq = iter([1.0, -1.0, 1.0, -1.0])
    monkeypatch.setattr(
        np.random,
        "default_rng",
        lambda: types.SimpleNamespace(choice=lambda _: next(seq)),
    )
    res = kw_gradient(lambda x: x * x, 1.0, h=0.5, n=4)
    # gradient of x^2 at x=1 is 2; with deterministic +- sequence it should match
    assert pytest.approx(res, rel=1e-6) == 2.0
