import networkx as nx

import datacreek.analysis.autotune as auto
from datacreek.analysis.autotune import AutoTuneState, autotune_step


def test_jitter_restart_counter(monkeypatch):
    state = AutoTuneState()
    state.stagnation = 4
    state.prev_costs = [0.0]
    g = nx.Graph()
    g.add_edge("a", "b")
    emb = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    labels = {"a": 0, "b": 1}
    monkeypatch.setattr(auto, "structural_entropy", lambda *a, **k: 0.0)
    monkeypatch.setattr(auto, "fractal_level_coverage", lambda *a, **k: 0.0)
    monkeypatch.setattr(auto, "bottleneck_distance", lambda *a, **k: 0.0)
    monkeypatch.setattr(auto, "graph_information_bottleneck", lambda *a, **k: 0.0)
    monkeypatch.setattr(auto, "mdl_description_length", lambda *a, **k: 0.0)
    monkeypatch.setattr(auto, "hybrid_score", lambda *a, **k: 0.0)
    called = {"n": 0}
    import datacreek.analysis.monitoring as monitoring

    monkeypatch.setattr(
        monitoring,
        "gp_jitter_restarts_total",
        type("C", (), {"inc": lambda self: called.__setitem__("n", called["n"] + 1)})(),
    )
    res = autotune_step(g, emb, labels, [g], state, penalty_cfg={"w_rec": 0.0})
    assert res["restart_gp"]
    assert called["n"] == 1
