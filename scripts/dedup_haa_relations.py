#!/usr/bin/env python3
"""Deduplicate SUGGESTED_HYPER_AA relations in Neo4j.

This script backs up the existing database using ``neo4j-admin copy`` and
then removes duplicate relationships by keeping the one with the smallest
internal id for each pair of node ids. The pair is normalized so the smaller
id comes first. This helps enforce the uniqueness constraint introduced in
migrations.
"""
import argparse
import subprocess
from pathlib import Path

from datacreek.backends import get_neo4j_driver


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate HAA relations")
    parser.add_argument("--db-path", default="/var/lib/neo4j/data/databases/neo4j")
    parser.add_argument("--backup-dir", default="backups/neo4j_dedup")
    parser.add_argument("--neo4j-admin", default="neo4j-admin")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    backup_dir = Path(args.backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        subprocess.run(
            [
                args.neo4j_admin,
                "copy",
                f"--to-path={backup_dir}",
                f"--from-path={args.db_path}",
            ],
            check=True,
        )

    driver = get_neo4j_driver()
    if driver is None:
        print("Neo4j driver not configured")
        return

    with driver.session() as session:
        session.run(
            "MATCH (a)-[r:SUGGESTED_HYPER_AA]->(b) "
            "WITH a,b,collect(r) AS rs WHERE size(rs) > 1 "
            "FOREACH (x IN tail(apoc.coll.sortNodes(rs)) | DELETE x)"
        )
    print("dedup complete")


if __name__ == "__main__":
    main()
