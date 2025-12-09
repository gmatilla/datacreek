#!/usr/bin/env python3
"""One-time script to create the HAA composite index."""

from datacreek.backends import get_neo4j_driver


def main() -> None:
    driver = get_neo4j_driver()
    if driver is None:
        print("Neo4j driver not configured")
        return
    with driver.session() as session:
        session.run(
            "CREATE INDEX haa_pair IF NOT EXISTS "
            "FOR ()-[r:SUGGESTED_HYPER_AA]-() "
            "ON (r.startNodeId, r.endNodeId)"
        )
    print("HAA index ensured")


if __name__ == "__main__":
    main()
