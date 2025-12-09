"""Tests for the edge review FastAPI UI."""

from __future__ import annotations

from fastapi.testclient import TestClient

from datacreek.edge_review import EDGE_REPAIR_LOG, EDGES, Edge, create_app


class DummyNeo4j:
    """Minimal stub capturing ``run`` calls."""

    def __init__(self) -> None:
        self.calls = []

    def run(
        self, tenant: str, cypher: str, **params
    ) -> None:  # pragma: no cover - trivial
        self.calls.append((tenant, cypher, params))


def setup_module(_) -> None:
    """Reset global stores before each test module."""

    EDGES.clear()
    EDGE_REPAIR_LOG.clear()


def test_list_and_patch_edge() -> None:
    """Edges above the threshold are listed and can be patched."""

    EDGES.extend(
        [
            Edge(id=1, src="a", dst="b", delta_lambda=0.2),
            Edge(id=2, src="a", dst="c", delta_lambda=0.05),
        ]
    )
    client_stub = DummyNeo4j()
    app = create_app(client_stub, tau=0.1)
    client = TestClient(app)

    # Only the first edge should appear due to delta_lambda > 0.1
    resp = client.get("/ui/edge_review")
    assert resp.status_code == 200
    assert [e["id"] for e in resp.json()] == [1]

    # Patching should invoke Neo4j and populate the repair log
    patch = client.patch("/ui/edge_review/1", json={"tenant": "t1", "action": "accept"})
    assert patch.status_code == 200
    assert client_stub.calls and client_stub.calls[0][2]["id"] == 1
    assert EDGE_REPAIR_LOG and EDGE_REPAIR_LOG[0]["action"] == "accept"
