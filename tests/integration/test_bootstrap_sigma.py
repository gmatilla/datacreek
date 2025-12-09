import types

import networkx as nx

from datacreek.analysis.fractal import bootstrap_sigma_db
from datacreek.core.fractal import bootstrap_sigma_db as core_bootstrap
from datacreek.core.knowledge_graph import KnowledgeGraph


def test_bootstrap_sigma_basic():
    g = nx.cycle_graph(6)
    sigma = bootstrap_sigma_db(g, [1])
    assert sigma >= 0.0
    assert "fractal_sigma" in g.graph


class DummySession:
    def __init__(self, log):
        self.log = log

    def run(self, query, **params):
        self.log.append(query)
        return types.SimpleNamespace(single=lambda: {"nodeIds": ["a", "b"]})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class DummyDriver:
    def __init__(self):
        self.log = []

    def session(self):
        return DummySession(self.log)


def test_core_bootstrap_uses_gds(monkeypatch):
    kg = KnowledgeGraph()
    kg.graph.add_edge("a", "b")
    monkeypatch.setattr(kg, "to_neo4j", lambda *a, **k: None)
    driver = DummyDriver()
    monkeypatch.setattr(
        "datacreek.core.fractal.colour_box_dimension", lambda g, r: (1.0, None)
    )
    sigma = core_bootstrap(kg, [1], driver=driver)
    assert sigma >= 0.0
    assert any("gds.beta.graph.sample" in q for q in driver.log)
