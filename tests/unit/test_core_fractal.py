import types

import networkx as nx

import datacreek.core.fractal as core_fractal


class DummyGraph:
    def __init__(self):
        self.graph = nx.path_graph(4)

    def set_property(self, key, value, driver=None, dataset=None):
        self.graph.graph[key] = value

    def to_neo4j(self, *a, **k):
        pass


def test_bootstrap_db_fallback(monkeypatch):
    kg = DummyGraph()
    # deterministic colour_box_dimension
    monkeypatch.setattr(
        core_fractal, "colour_box_dimension", lambda *_a, **_k: (1.0, [])
    )
    monkeypatch.setattr(core_fractal, "update_metric", lambda *a, **k: None)
    monkeypatch.setattr(
        "datacreek.utils.config.load_config",
        lambda: {"fractal": {"bootstrap_seed": 42}},
    )
    dims = core_fractal.bootstrap_db(kg, n=3, ratio=0.5, driver=None)
    assert len(dims) == 3
    assert kg.graph.graph["fractal_dim"] == 1.0
    assert kg.graph.graph["fractal_sigma"] == 0.0


def test_bootstrap_sigma_db(monkeypatch):
    kg = DummyGraph()
    monkeypatch.setattr(
        core_fractal, "colour_box_dimension", lambda *_a, **_k: (1.0, [])
    )
    monkeypatch.setattr(core_fractal, "update_metric", lambda *a, **k: None)
    sigma = core_fractal.bootstrap_sigma_db(kg, radii=[1], driver=None)
    assert sigma == 0.0
    assert kg.graph.graph["fractal_sigma"] == 0.0


def test_bootstrap_db_error_handling(monkeypatch):
    kg = DummyGraph()
    monkeypatch.setattr(
        core_fractal,
        "colour_box_dimension",
        lambda *_a, **_k: (1.0, []),
    )
    monkeypatch.setattr(
        "datacreek.utils.config.load_config", lambda: {"fractal": {"bootstrap_seed": 0}}
    )

    class FailingMetric:
        def __call__(self, *a, **k):
            raise RuntimeError

    monkeypatch.setattr(core_fractal, "update_metric", FailingMetric())
    monkeypatch.setattr(core_fractal, "Driver", None)
    dims = core_fractal.bootstrap_db(kg, n=2, ratio=0.5, driver=None)
    assert len(dims) == 2


class DummyResult:
    def single(self):
        return {"nodeIds": [0, 1]}

    def get(self, name, default=None):
        return self.single().get(name, default)


class DummySession:
    def __init__(self):
        self.queries = []

    def run(self, query, **kwargs):
        self.queries.append(query)
        if "RETURN a.id AS u" in query:
            return [{"u": 0, "v": 1}]
        return DummyResult()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class DummyDriver:
    def session(self):
        return DummySession()


def test_bootstrap_db_with_driver(monkeypatch):
    kg = DummyGraph()
    monkeypatch.setattr(
        core_fractal, "colour_box_dimension", lambda *_a, **_k: (1.0, [])
    )
    monkeypatch.setattr(core_fractal, "update_metric", lambda *a, **k: None)
    monkeypatch.setattr(
        "datacreek.utils.config.load_config", lambda: {"fractal": {"bootstrap_seed": 0}}
    )
    driver = DummyDriver()
    dims = core_fractal.bootstrap_db(kg, n=2, ratio=0.5, driver=driver, dataset="d")
    assert len(dims) == 2
    assert kg.graph.graph["fractal_dim"] == 1.0


def test_bootstrap_sigma_db_with_driver(monkeypatch):
    kg = DummyGraph()
    monkeypatch.setattr(
        core_fractal, "colour_box_dimension", lambda *_a, **_k: (1.0, [])
    )
    monkeypatch.setattr(core_fractal, "update_metric", lambda *a, **k: None)
    monkeypatch.setattr(
        "datacreek.utils.config.load_config", lambda: {"fractal": {"bootstrap_seed": 0}}
    )
    driver = DummyDriver()
    sigma = core_fractal.bootstrap_sigma_db(kg, radii=[1], driver=driver, dataset="d")
    assert sigma == 0.0
    assert kg.graph.graph["fractal_dim"] == 1.0
