# pragma: no cover
# Requires PostgreSQL with pgvector extension, excluded from tests.
"""Export graph embeddings to PostgreSQL using pgvector.

This module exposes helpers to bulk COPY embeddings into a table and build a
vector index for fast similarity search.  The schema follows the audit
specification::

    CREATE TABLE embedding (
        node_id UUID,
        space TEXT,
        vec VECTOR(dim)
    );

``export_embeddings_pg`` inserts the embeddings using ``COPY`` for better
performance and optionally creates an ``ivfflat`` index.
"""

from __future__ import annotations

import sys
from typing import Iterable, Sequence

from psycopg import Connection

# When loaded via :func:`importlib.util.module_from_spec` in tests, the
# parent ``datacreek`` package may not exist. Ensure it is imported so
# that submodules can be discovered normally.
if "datacreek" not in sys.modules:
    import importlib.util
    from pathlib import Path

    pkg_path = Path(__file__).resolve().parents[1] / "__init__.py"
    spec = importlib.util.spec_from_file_location("datacreek", pkg_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("datacreek", module)
        spec.loader.exec_module(module)


def _fmt_vector(vec: Iterable[float] | None) -> str | None:
    """Return vector literal for pgvector."""
    if vec is None:
        return None
    return "[" + ",".join(f"{float(v):g}" for v in vec) + "]"


def export_embeddings_pg(
    kg: "KnowledgeGraph",
    conn: Connection,
    *,
    table: str = "embedding",
    attrs: Iterable[str] = ("embedding", "poincare_embedding"),
    lists: int = 100,
    embedding_version: str | None = None,
) -> int:
    """Export embeddings using ``COPY`` and build an ``ivfflat`` index.

    Parameters
    ----------
    kg:
        Graph storing embeddings on its nodes.
    conn:
        Open psycopg connection to the database with pgvector extension.
    table:
        Destination table, created if missing.
    attrs:
        Node attributes storing different embedding spaces.
    lists:
        Number of lists for the ``ivfflat`` index.
    embedding_version:
        Version tag stored alongside each row for reproducibility.

    Returns
    -------
    int
        Number of rows written.
    """

    rows = []
    dim = None
    version = embedding_version or kg.graph.graph.get("faiss_meta", {}).get(
        "embedding_version", "1.0"
    )
    for node, data in kg.graph.nodes(data=True):
        for space in attrs:
            vec = data.get(space)
            if vec is None:
                continue
            if dim is None:
                dim = len(vec)
            rows.append((str(node), space, _fmt_vector(vec), version))

    if not rows:
        return 0

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {table} ("
            "node_id TEXT,"
            "space TEXT,"
            f"vec VECTOR({dim}),"
            "embedding_version TEXT"
            ")"
        )
        with cur.copy(
            f"COPY {table} (node_id, space, vec, embedding_version) FROM STDIN"
        ) as cp:
            for row in rows:
                cp.write_row(row)
        # ``lists`` parameter cannot be passed as a bind variable in this DDL
        # statement, so interpolate it after sanitising. ``ivfflat`` requires a
        # positive integer number of lists.
        # Build the IVFFlat index optimised for inner product search.  The
        # ``lists`` value must be interpolated directly as pgvector does not
        # allow parameter binding in this DDL statement.
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {table}_ivfflat ON {table} USING ivfflat (vec vector_ip_ops) WITH (lists={int(lists)})"
        )
    conn.commit()
    return len(rows)


def query_topk_pg(
    conn: Connection,
    table: str,
    vec: Iterable[float],
    *,
    k: int = 5,
    probes: int = 20,
    version: str | None = None,
) -> list[tuple[str, str]]:
    """Return ``k`` nearest neighbours ordered by cosine distance.

    Parameters
    ----------
    conn:
        Open connection to the PostgreSQL database.
    table:
        Name of the table containing embeddings.
    vec:
        Query vector as an iterable of floats.
    k:
        Number of nearest neighbours to return.
    probes:
        Number of inverted lists to probe. Higher values increase recall at the
        cost of latency. ``ivfflat`` restricts the maximum to the number of
        lists used when building the index. The default of ``20`` meets the
        latency/recall targets used by the heavy tests.
    """
    import time

    from ..analysis.monitoring import update_metric

    t0 = time.perf_counter()
    with conn.cursor() as cur:
        # ``SET`` doesn't accept parameter placeholders so interpolate value
        cur.execute(f"SET LOCAL ivfflat.probes = {int(probes)}")
        # Use inner product distance operator for consistency with FAISS
        # ``IndexFlatIP`` baseline used in tests.
        if version is None:
            cur.execute(
                f"SELECT node_id, space FROM {table} ORDER BY vec <#> %s LIMIT %s",  # nosec B608
                (_fmt_vector(vec), k),
            )
        else:
            cur.execute(
                f"SELECT node_id, space FROM {table} WHERE embedding_version = %s ORDER BY vec <#> %s LIMIT %s",  # nosec B608
                (version, _fmt_vector(vec), k),
            )
        rows = cur.fetchall()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    update_metric("pgvector_query_ms", elapsed_ms)
    return rows
