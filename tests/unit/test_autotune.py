import networkx as nx

from datacreek.analysis.autotune import AutoTuneState, recall_at_k, update_theta


class DummyGauge:
    def __init__(self):
        self.value = None

    def set(self, v):
        self.value = v


def test_recall_at_k_basic():
    """Hybrid retrieval should hit the relevant node."""
    graph = nx.Graph()
    graph.add_node(
        "q",
        embedding=[1.0, 0.0],
        graphwave_embedding=[0.5, 0.5],
        poincare_embedding=[0.2, 0.2],
    )
    graph.add_node(
        "a",
        embedding=[1.0, 0.0],
        graphwave_embedding=[0.5, 0.5],
        poincare_embedding=[0.2, 0.2],
    )
    graph.add_node(
        "b",
        embedding=[0.0, 1.0],
        graphwave_embedding=[0.5, -0.5],
        poincare_embedding=[-0.2, -0.2],
    )

    r = recall_at_k(graph, ["q"], {"q": ["a"]}, k=1)
    assert r == 1.0


def test_recall_at_k_missing_embeddings():
    """Queries lacking embeddings yield zero recall."""
    graph = nx.Graph()
    graph.add_node("q", embedding=None)

    r = recall_at_k(graph, ["q"], {"q": ["a"]}, k=1)
    assert r == 0.0


def test_update_theta(monkeypatch):
    """update_theta should record the cost and update parameters."""
    gauge = DummyGauge()
    import datacreek.analysis.monitoring as monitoring

    monkeypatch.setattr(monitoring, "j_cost", gauge, raising=False)

    state = AutoTuneState()
    metrics = {
        "cost": 3.14,
        "tau": 5,
        "eps": 0.1,
        "beta": 0.2,
        "delta": 0.3,
        "jitter": 0.5,
    }
    update_theta(state, metrics)

    assert gauge.value == 3.14
    assert state.tau == 5
    assert state.eps == 0.1
    assert state.beta == 0.2
    assert state.delta == 0.3
    assert state.jitter == 0.5
