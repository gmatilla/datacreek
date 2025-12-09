from datacreek.core.knowledge_graph import KnowledgeGraph


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


def test_set_property_persists(monkeypatch):
    kg = KnowledgeGraph()
    driver = DummyDriver()
    kg.set_property("foo", 1.23, driver=driver, dataset="d")
    assert kg.graph.graph["foo"] == 1.23
    assert any("foo" in q for q in driver.log)
