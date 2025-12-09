from datacreek.core.knowledge_graph import KnowledgeGraph


def test_subgraph_fractal_dimension(monkeypatch):
    kg = KnowledgeGraph()
    kg.add_document("d", "src")
    kg.add_chunk("d", "c1", "a")
    kg.add_chunk("d", "c2", "b")

    monkeypatch.setattr(
        "datacreek.analysis.fractal.box_counting_dimension",
        lambda g, r: (1.2, [(1, 2)]),
    )

    dim = kg.subgraph_fractal_dimension(["c1", "c2"], radii=[1])
    assert dim == 1.2
    assert kg.graph.nodes["c1"]["fractal_dim"] == 1.2
    assert kg.graph.nodes["c2"]["fractal_dim"] == 1.2


class DummySession:
    def __init__(self):
        self.queries = []

    def run(self, query, **kw):
        self.queries.append(query)
        return []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class DummyDriver:
    def __init__(self):
        self.session_obj = DummySession()

    def session(self):
        return self.session_obj


def test_create_fractal_index():
    kg = KnowledgeGraph()
    driver = DummyDriver()
    kg.create_fractal_index(driver)
    assert any("createNodeIndex" in q for q in driver.session_obj.queries)
