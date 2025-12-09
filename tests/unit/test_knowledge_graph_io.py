import types

from datacreek.core.knowledge_graph import KnowledgeGraph


class DummySession:
    def __init__(self):
        self.queries = []
        self.read_nodes = []
        self.read_edges = []

    def run(self, query, **params):
        self.queries.append(query)
        if "RETURN n, labels(n)[0] AS label" in query:
            return [
                {
                    "n": {"id": "d1", "type": "document", "uid": None},
                    "label": "Document",
                },
                {"n": {"id": "c1", "type": "chunk", "text": "hello"}, "label": "Chunk"},
            ]
        if "RETURN a.id AS src" in query:
            return [{"src": "d1", "rel": "HAS_CHUNK", "tgt": "c1", "rel_props": {}}]
        return []

    def execute_write(self, fn):
        fn(self)

    def execute_read(self, fn):
        fn(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class DummyDriver:
    def __init__(self):
        self.session_obj = DummySession()

    def session(self):
        return self.session_obj


def test_to_from_neo4j_roundtrip():
    kg = KnowledgeGraph()
    kg.add_document("d1", "src")
    kg.add_chunk("d1", "c1", "hello")
    driver = DummyDriver()
    kg.to_neo4j(driver, dataset="ds")
    # the first MERGE statement for documents should include timestamp fields
    merged = "".join(driver.session_obj.queries)
    assert "first_seen" in merged
    assert "last_ingested" in merged
    kg2 = KnowledgeGraph.from_neo4j(driver, dataset="ds")
    assert kg2.graph.nodes["d1"]["type"] == "document"
    assert kg2.graph.nodes["c1"].get("text") == "hello"
    assert ("d1", "c1") in kg2.graph.edges
