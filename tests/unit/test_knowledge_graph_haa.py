from datacreek.core.knowledge_graph import KnowledgeGraph


class DummyRecord:
    def __init__(self, score=None):
        self.score = score

    def __getitem__(self, key):
        return self.score


class DummyResult:
    def __init__(self, score=None):
        self.score = score

    def single(self):
        return DummyRecord(self.score) if self.score is not None else None


class DummySession:
    def __init__(self, score=None):
        self.score = score
        self.queries = []

    def run(self, query, **params):
        self.queries.append((query, params))
        return DummyResult(self.score)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class DummyDriver:
    def __init__(self, score=None):
        self.session_obj = DummySession(score)

    def session(self):
        return self.session_obj


def test_haa_link_score():
    kg = KnowledgeGraph()
    driver = DummyDriver(score=0.8)
    score = kg.haa_link_score(driver, "a", "b")
    assert score == 0.8
    assert driver.session_obj.queries

    miss_driver = DummyDriver(score=None)
    assert kg.haa_link_score(miss_driver, "a", "b") is None
