import os

import pytest

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - optional dependency
    GraphDatabase = None  # type: ignore

pytest.importorskip("pydantic")

from datacreek.core.knowledge_graph import KnowledgeGraph


def get_driver():
    if GraphDatabase is None:
        pytest.skip("neo4j package not installed")
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    password = os.environ.get("NEO4J_PASSWORD")
    if not uri or not user or not password:
        pytest.skip("Neo4j not configured")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
    except Exception:
        pytest.skip("Neo4j server unavailable")
    return driver


def test_knowledge_graph_roundtrip():
    driver = get_driver()
    kg = KnowledgeGraph()
    kg.add_document("d1", source="s", uid="hash1")
    kg.add_chunk("d1", "c1", "hello")
    kg.to_neo4j(driver, clear=True)
    loaded = KnowledgeGraph.from_neo4j(driver)
    driver.close()
    assert loaded.search_chunks("hello") == ["c1"]


def test_merge_by_uid():
    driver = get_driver()
    kg = KnowledgeGraph()
    kg.add_document("d1", source="s", uid="hash1")
    kg.add_chunk("d1", "c1", "hello")
    kg.to_neo4j(driver, clear=True)
    kg.to_neo4j(driver, clear=False)
    with driver.session() as session:
        count = session.run(
            "MATCH (d:Document {uid:$uid}) RETURN count(d)",
            uid="hash1",
        ).single()[0]
    driver.close()
    assert count == 1
