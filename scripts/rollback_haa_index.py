#!/usr/bin/env python3
"""Rollback the HAA migration by removing index and constraint."""
from datacreek.backends import get_neo4j_driver


def main() -> None:
    driver = get_neo4j_driver()
    if driver is None:
        print("Neo4j driver not configured")
        return
    with driver.session() as session:
        session.run("DROP CONSTRAINT haa_pair_unique IF EXISTS")
        session.run("DROP INDEX haa_pair IF EXISTS")
    print("HAA index and constraint removed")


if __name__ == "__main__":
    main()
