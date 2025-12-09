#!/usr/bin/env python3
"""Normalize HAA relationship orientation and output checksum."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[0] = str(ROOT)

import os

os.environ.setdefault("DATACREEK_REQUIRE_PERSISTENCE", "0")

from datacreek.backends import get_neo4j_driver


def main() -> None:
    driver = get_neo4j_driver()
    if driver is None:
        print("Neo4j driver not configured")
        return
    with driver.session() as session:
        session.run(
            "MATCH (a)-[r:SUGGESTED_HYPER_AA]->(b) "
            "WITH CASE WHEN r.startNodeId <= r.endNodeId THEN r.startNodeId ELSE r.endNodeId END AS s, "
            "CASE WHEN r.startNodeId <= r.endNodeId THEN r.endNodeId ELSE r.startNodeId END AS t, r "
            "SET r.startNodeId = s, r.endNodeId = t"
        )
        session.run(
            "MATCH ()-[r:SUGGESTED_HYPER_AA]->() "
            "WITH r.startNodeId AS s, r.endNodeId AS t, collect(r) AS rs "
            "WHERE size(rs) > 1 "
            "FOREACH (x IN tail(rs) | DELETE x)"
        )
        rec = session.run(
            "MATCH ()-[r:SUGGESTED_HYPER_AA]->() "
            "RETURN count(r) AS c, sum(r.startNodeId + r.endNodeId) AS sum"
        ).single()
    checksum = f"{rec['c']}-{rec['sum']}" if rec else "0-0"
    print(checksum)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
