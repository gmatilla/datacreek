import os
import tempfile
import types

import networkx as nx
import pytest

from datacreek.core import fractal as core_fractal
from datacreek.core.knowledge_graph import KnowledgeGraph


@pytest.mark.heavy
def test_bootstrap_db_and_sigma(monkeypatch):
    g = KnowledgeGraph()
    g.graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    # create minimal config file
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp.write("fractal:\n  bootstrap_seed: 1\n")
        cfg_path = tmp.name
    monkeypatch.setenv("DATACREEK_CONFIG", cfg_path)
    monkeypatch.setattr(core_fractal, "update_metric", lambda *a, **k: None)
    dims = core_fractal.bootstrap_db(g, n=3, ratio=0.5)
    assert len(dims) == 3
    assert g.graph.graph["fractal_seed"] == 1
    sigma = core_fractal.bootstrap_sigma_db(g, [1, 2])
    assert isinstance(sigma, float)
    assert g.graph.graph.get("fractal_sigma") == sigma
    os.remove(cfg_path)


class DummySession:
    def __init__(self):
        self.calls = []

    def run(self, query, **kwargs):
        self.calls.append(query)
        if "sample" in query:
            return types.SimpleNamespace(single=lambda: {"nodeIds": [0, 1]})
        if "RETURN a.id AS u" in query:
            return [{"u": 0, "v": 1}]
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class DummyDriver:
    def __init__(self):
        self.sess = DummySession()

    def session(self):
        return self.sess


@pytest.mark.heavy
def test_bootstrap_db_with_driver(monkeypatch):
    g = KnowledgeGraph()
    g.graph.add_edge(0, 1)
    driver = DummyDriver()
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp.write("fractal:\n  bootstrap_seed: 2\n")
        cfg_path = tmp.name
    monkeypatch.setenv("DATACREEK_CONFIG", cfg_path)
    monkeypatch.setattr(core_fractal, "update_metric", lambda *a, **k: None)
    monkeypatch.setattr(
        KnowledgeGraph, "to_neo4j", lambda self, driver, dataset, clear: None
    )
    dims = core_fractal.bootstrap_db(g, n=1, ratio=1.0, driver=driver, dataset="ds")
    assert dims
    assert "fractal_dim" in g.graph.graph
    os.remove(cfg_path)


@pytest.mark.heavy
def test_bootstrap_db_empty(monkeypatch):
    g = KnowledgeGraph()
    g.graph.add_node(0)
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp.write("fractal:\n  bootstrap_seed: 3\n")
        cfg_path = tmp.name
    monkeypatch.setenv("DATACREEK_CONFIG", cfg_path)
    monkeypatch.setattr(core_fractal, "update_metric", lambda *a, **k: None)
    dims = core_fractal.bootstrap_db(g, n=0, ratio=0.5)
    assert dims == [0.0] or dims == [0] or len(dims) == 1
    os.remove(cfg_path)


@pytest.mark.heavy
def test_bootstrap_sigma_with_driver(monkeypatch):
    g = KnowledgeGraph()
    g.graph.add_edges_from([(0, 1), (1, 2)])
    driver = DummyDriver()
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp.write("fractal:\n  bootstrap_seed: 4\n")
        cfg_path = tmp.name
    monkeypatch.setenv("DATACREEK_CONFIG", cfg_path)
    monkeypatch.setattr(core_fractal, "update_metric", lambda *a, **k: None)
    monkeypatch.setattr(
        KnowledgeGraph, "to_neo4j", lambda self, driver, dataset, clear: None
    )
    sigma = core_fractal.bootstrap_sigma_db(g, [1], driver=driver, dataset="d")
    assert isinstance(sigma, float)
    os.remove(cfg_path)
