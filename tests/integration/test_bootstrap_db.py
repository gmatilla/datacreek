from datacreek.core import fractal as fractal_mod
from datacreek.core.knowledge_graph import KnowledgeGraph

bootstrap_db = fractal_mod.bootstrap_db


def test_bootstrap_db_records_properties():
    kg = KnowledgeGraph()
    kg.graph.add_edge("a", "b")
    dims = bootstrap_db(kg, n=3, ratio=0.8)
    assert len(dims) == 3
    assert "fractal_dim" in kg.graph.graph
    assert "fractal_sigma" in kg.graph.graph


class DummySession:
    def __init__(self, log):
        self.log = log

    def run(self, query, **params):
        self.log.append(query)
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class DummyDriver:
    def __init__(self):
        self.log = []

    def session(self):
        return DummySession(self.log)


def test_bootstrap_db_writes_neo4j(monkeypatch):
    kg = KnowledgeGraph()
    kg.graph.add_edge("a", "b")
    monkeypatch.setattr(kg, "to_neo4j", lambda *a, **k: None)
    driver = DummyDriver()
    monkeypatch.setattr(
        "datacreek.core.fractal.colour_box_dimension", lambda g, r: (1.0, None)
    )
    bootstrap_db(kg, n=2, ratio=0.5, driver=driver, dataset="tmp")
    assert any("GraphMeta" in q for q in driver.log)
    assert any("fractal_dim" in q for q in driver.log)
    assert any("fractal_sigma" in q for q in driver.log)


def test_bootstrap_db_records_seed(monkeypatch):
    kg = KnowledgeGraph()
    kg.graph.add_edge("a", "b")
    called = []
    monkeypatch.setattr("numpy.random.seed", lambda s: called.append(s))
    monkeypatch.setattr(kg, "to_neo4j", lambda *a, **k: None)
    driver = DummyDriver()
    monkeypatch.setattr(
        "datacreek.core.fractal.colour_box_dimension", lambda g, r: (1.0, None)
    )
    bootstrap_db(kg, n=1, ratio=0.5, driver=driver, dataset="tmp")
    assert called
    assert any("fractal_seed" in q for q in driver.log)
