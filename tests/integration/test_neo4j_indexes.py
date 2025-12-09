from datacreek.backends import ensure_neo4j_indexes, run_cypher_file


class DummySession:
    def __init__(self):
        self.queries = []

    def run(self, query):
        self.queries.append(query)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class DummyDriver:
    def __init__(self):
        self.session_obj = DummySession()

    def session(self):
        return self.session_obj


def test_haa_index_migration():
    driver = DummyDriver()
    ensure_neo4j_indexes(driver)
    assert any("CREATE INDEX haa_pair" in q for q in driver.session_obj.queries)


def test_run_cypher_file(tmp_path):
    file = tmp_path / "m.cypher"
    file.write_text("CREATE INDEX foo IF NOT EXISTS FOR (n:Doc) ON (n.id);")
    driver = DummyDriver()
    run_cypher_file(driver, str(file))
    assert any("CREATE INDEX foo" in q for q in driver.session_obj.queries)
