"""Unit tests for the Neo4j Fabric client utilities."""

from pathlib import Path
from typing import Any, Dict

from datacreek.neo4j_fabric import Neo4jFabricClient, flyway_migration_path


class FakeResult:
    """Minimal stand-in for ``neo4j.Result``."""

    def __init__(self, records):
        self._records = records

    def data(self):  # pragma: no cover - trivial
        return self._records


class FakeSession:
    def __init__(self, store: Dict[str, list[str]]):
        self.store = store

    def run(self, statement: str, **params: Any) -> FakeResult:
        tokens = statement.split()
        assert tokens[0] == "USING" and tokens[1] == "DATABASE"
        tenant = tokens[2]
        cypher = " ".join(tokens[3:])
        if cypher.startswith("CREATE"):
            self.store.setdefault(tenant, []).append(params["name"])
            return FakeResult([])
        if cypher.startswith("MATCH"):
            records = [{"name": n} for n in self.store.get(tenant, [])]
            return FakeResult(records)
        raise ValueError("Unsupported Cypher")

    # context manager protocol
    def __enter__(self):  # pragma: no cover - trivial
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - trivial
        return False


class FakeDriver:
    def __init__(self):
        self.store: Dict[str, list[str]] = {}

    def session(self):  # pragma: no cover - trivial
        return FakeSession(self.store)


def test_queries_are_isolated_between_tenants():
    driver = FakeDriver()
    client = Neo4jFabricClient(driver)

    client.run("tenantA", "CREATE (:User {name:$name})", name="Alice")
    client.run("tenantB", "CREATE (:User {name:$name})", name="Bob")

    data_a = client.run("tenantA", "MATCH (u:User) RETURN u.name AS name").data()
    data_b = client.run("tenantB", "MATCH (u:User) RETURN u.name AS name").data()

    assert data_a == [{"name": "Alice"}]
    assert data_b == [{"name": "Bob"}]


def test_flyway_migration_path():
    assert flyway_migration_path("alpha") == Path("db/migration/alpha")
