"""Utilities for interacting with Neo4j Fabric in a multi-tenant setup.

This module provides a lightweight client that ensures every Cypher query is
routed to a tenant-specific database using the ``USING DATABASE`` clause. It
also exposes a helper to compute the Flyway migration path for a given tenant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # optional neo4j dependency
    from neo4j import Driver, Result  # type: ignore
except Exception:  # pragma: no cover - dependency not installed for tests
    Driver = Any  # type: ignore
    Result = Any  # type: ignore


@dataclass
class Neo4jFabricClient:
    """Client wrapper that routes queries to a tenant database in Neo4j Fabric."""

    driver: Driver

    def run(self, tenant: str, cypher: str, **params: Any) -> Result:
        """Execute a Cypher statement against ``tenant``'s database.

        Parameters
        ----------
        tenant:
            Identifier of the tenant's Fabric database.
        cypher:
            Cypher query to execute *without* a ``USING DATABASE`` clause.
        params:
            Parameters forwarded to the Neo4j driver.

        Returns
        -------
        Result
            The result returned by :mod:`neo4j`.
        """
        statement = f"USING DATABASE {tenant} {cypher}"
        # ``session`` implements the context manager protocol on the real driver
        with self.driver.session() as session:  # type: ignore[union-attr]
            return session.run(statement, **params)  # type: ignore[arg-type]


def flyway_migration_path(tenant: str) -> Path:
    """Return the Flyway migration directory for ``tenant``.

    Examples
    --------
    >>> flyway_migration_path("alpha")
    PosixPath('db/migration/alpha')
    """
    return Path("db") / "migration" / tenant


# Supported hyperedge types for multiplexed graphs.
EDGE_TYPES = ("DOC", "USER", "TAG")


def extend_schema_with_edge_labels(client: Neo4jFabricClient, tenant: str) -> None:
    """Ensure typed hyperedge labels and constraints exist in Neo4j.

    For every hyperedge type ``t`` in :data:`EDGE_TYPES`, this helper performs two
    operations on the tenant's database:

    1. Relabel existing ``:EDGE`` nodes that declare ``type = t`` with the more
       specific label ``:EDGE_{t}``.  This mirrors the typed-edge notion used in
       the in-memory hypergraph and enables type-selective queries.
    2. Create a uniqueness constraint on the ``id`` property for nodes carrying
       that label so that hyperedges can be addressed efficiently.

    Parameters
    ----------
    client:
        Neo4j client used to execute Cypher statements.
    tenant:
        Identifier of the tenant database in Fabric.
    """

    for edge_type in EDGE_TYPES:
        label = f"EDGE_{edge_type}"
        # ``MATCH`` is used to update existing :EDGE nodes with a more specific
        # label so downstream queries can select by type without filtering.
        client.run(
            tenant,
            f"MATCH (e:EDGE {{type: '{edge_type}'}}) SET e:{label}",
        )
        # Enforce uniqueness on the ``id`` property for the newly labelled nodes
        # to keep hyperedge identifiers collision-free.
        client.run(
            tenant,
            (
                f"CREATE CONSTRAINT edge_{edge_type.lower()}_id IF NOT EXISTS "
                f"FOR (e:{label}) REQUIRE e.id IS UNIQUE"
            ),
        )
