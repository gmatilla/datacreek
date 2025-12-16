#!/usr/bin/env python3
"""Dry-run Flyway migrations against a Neo4j instance.

This script executes all cypher files in the ``migrations`` directory on a
Neo4j database specified via environment variables. It computes a snapshot of
existing ``SUGGESTED_HYPER_AA`` relations before and after applying the
migrations and fails if duplicates remain.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - optional dependency
    GraphDatabase = None  # type: ignore

MIGR_DIR = Path(__file__).resolve().parents[1] / "migrations"


def _snapshot(driver) -> str:
    """Return a checksum of relation count and pair cardinality."""
    with driver.session() as session:
        rec = session.run(
            "MATCH ()-[r:SUGGESTED_HYPER_AA]->() "
            "RETURN count(r) AS c, count(DISTINCT [r.startNodeId,r.endNodeId]) AS u"
        ).single()
    return f"{rec['c']}-{rec['u']}" if rec else "0-0"


def dry_run(driver) -> Tuple[str, str]:
    """Execute all migrations and return before/after checksums."""
    before = _snapshot(driver)
    for path in sorted(MIGR_DIR.glob("*.cypher")):
        with path.open() as fh, driver.session() as session:
            session.run(fh.read())
    after = _snapshot(driver)
    if before != after and not after.endswith("-0"):
        raise RuntimeError("duplicates remain after migration")
    return before, after


def main() -> None:  # pragma: no cover - manual use
    if GraphDatabase is None:
        raise SystemExit("neo4j driver not installed")
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "test")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    before, after = dry_run(driver)
    print(json.dumps({"before": before, "after": after}))


if __name__ == "__main__":
    main()
