import datacreek.core.knowledge_graph as kgmod
from datacreek.core.knowledge_graph import KnowledgeGraph


class _Sess:
    def __init__(self):
        self.queries = []

    def run(self, query, **params):
        self.queries.append(query)

        class _Rec:
            def single(self):
                return {"score": 1.5}

        return _Rec()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class _Drv:
    def __init__(self):
        self.session_obj = _Sess()

    def session(self):
        return self.session_obj


def test_haa_index_query_plan():
    kg = KnowledgeGraph()
    drv = _Drv()
    score = kg.haa_link_score(drv, "u", "v")
    assert score == 1.5
    q = drv.session_obj.queries[-1]
    assert q.startswith("PROFILE MATCH")
    assert "CASE WHEN ida <= idb" in q
