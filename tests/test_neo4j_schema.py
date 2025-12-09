"""Tests for Neo4j schema extension with typed edge labels."""

from __future__ import annotations

from datacreek.neo4j_fabric import extend_schema_with_edge_labels


class DummyClient:
    """Capture Cypher calls for verification."""

    def __init__(self) -> None:
        self.calls = []

    def run(
        self, tenant: str, cypher: str, **params
    ) -> None:  # pragma: no cover - trivial
        self.calls.append((tenant, cypher, params))


def test_extend_schema_with_edge_labels() -> None:
    """Edge labels and constraints are created for all supported types."""

    client = DummyClient()
    extend_schema_with_edge_labels(client, tenant="alpha")

    assert client.calls == [
        ("alpha", "MATCH (e:EDGE {type: 'DOC'}) SET e:EDGE_DOC", {}),
        (
            "alpha",
            "CREATE CONSTRAINT edge_doc_id IF NOT EXISTS FOR (e:EDGE_DOC) REQUIRE e.id IS UNIQUE",
            {},
        ),
        ("alpha", "MATCH (e:EDGE {type: 'USER'}) SET e:EDGE_USER", {}),
        (
            "alpha",
            "CREATE CONSTRAINT edge_user_id IF NOT EXISTS FOR (e:EDGE_USER) REQUIRE e.id IS UNIQUE",
            {},
        ),
        ("alpha", "MATCH (e:EDGE {type: 'TAG'}) SET e:EDGE_TAG", {}),
        (
            "alpha",
            "CREATE CONSTRAINT edge_tag_id IF NOT EXISTS FOR (e:EDGE_TAG) REQUIRE e.id IS UNIQUE",
            {},
        ),
    ]
