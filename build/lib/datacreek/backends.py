import os

try:  # optional dependency
    import boto3
except Exception:  # pragma: no cover - handled gracefully for tests
    boto3 = None
import os
from functools import lru_cache
from pathlib import Path

try:  # optional Redis dependency
    import redis
except Exception:  # pragma: no cover - optional dependency missing
    redis = None  # type: ignore

try:  # optional Neo4j dependency
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - optional dependency missing
    GraphDatabase = None  # type: ignore

from datacreek.storage import S3Storage
from datacreek.utils.config import (
    get_neo4j_config,
    get_redis_config,
    load_config_with_overrides,
)

try:  # optional dependency
    from redisgraph import Graph as RedisGraph
except Exception:  # pragma: no cover - optional
    RedisGraph = None


@lru_cache()
def get_redis_client(config_path: str | None = None):
    """Return a Redis client configured via config file or environment."""

    if redis is None:
        return None
    cfg = get_redis_config(load_config_with_overrides(config_path))
    host = os.getenv("REDIS_HOST", cfg.get("host"))
    port = int(os.getenv("REDIS_PORT", cfg.get("port", 6379)))
    return redis.Redis(host=host, port=port, decode_responses=True)


@lru_cache()
def get_neo4j_driver(config_path: str | None = None):
    """Return a Neo4j driver configured via config file or environment."""
    if GraphDatabase is None:
        return None

    cfg = get_neo4j_config(load_config_with_overrides(config_path))
    uri = os.getenv("NEO4J_URI", cfg.get("uri"))
    user = os.getenv("NEO4J_USER", cfg.get("user"))
    password = os.getenv("NEO4J_PASSWORD", cfg.get("password"))
    if not uri or not user or not password:
        return None
    driver = GraphDatabase.driver(uri, auth=(user, password))
    if os.getenv("NEO4J_INIT_INDEXES", "1") != "0":
        ensure_neo4j_indexes(driver)
    if cfg.get("run_migrations"):
        run_cypher_file(driver, "2025-07-haa_index.cypher")
        run_cypher_file(driver, "2025-07-haa_unique_constraint.cypher")
    return driver


def ensure_neo4j_indexes(driver) -> None:
    """Create indexes used by the knowledge graph if missing."""

    statements = [
        "CREATE INDEX document_id IF NOT EXISTS FOR (n:Document) ON (n.id)",
        "CREATE INDEX section_id IF NOT EXISTS FOR (n:Section) ON (n.id)",
        "CREATE INDEX chunk_id IF NOT EXISTS FOR (n:Chunk) ON (n.id)",
        "CREATE INDEX image_id IF NOT EXISTS FOR (n:Image) ON (n.id)",
        "CREATE INDEX entity_id IF NOT EXISTS FOR (n:Entity) ON (n.id)",
        "CREATE INDEX fact_id IF NOT EXISTS FOR (n:Fact) ON (n.id)",
        "CREATE INDEX dataset_nodes IF NOT EXISTS FOR (n) ON (n.dataset)",
        "CREATE INDEX dataset_rels IF NOT EXISTS FOR ()-[r]-() ON (r.dataset)",
        (
            "CREATE INDEX haa_pair IF NOT EXISTS "
            "FOR ()-[r:SUGGESTED_HYPER_AA]-() "
            "ON (r.startNodeId, r.endNodeId)"
        ),
    ]

    with driver.session() as session:
        for stmt in statements:
            try:
                session.run(stmt)
            except Exception:
                # index creation is best-effort
                continue


def run_cypher_file(driver, filename: str) -> None:
    """Execute Cypher statements from ``filename`` if it exists."""

    path = Path(__file__).resolve().parent.parent / "migrations" / filename
    if not path.exists():
        return
    with open(path) as fh, driver.session() as session:
        for stmt in fh.read().split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    session.run(stmt)
                except Exception:
                    continue


@lru_cache()
def get_redis_graph(name: str, config_path: str | None = None):
    """Return a RedisGraph instance for ``name`` if available."""

    if RedisGraph is None:
        return None
    if os.getenv("USE_REDIS_GRAPH", "0") in {"0", "false", "False"}:
        return None
    cfg = get_redis_config(load_config_with_overrides(config_path))
    host = os.getenv("REDIS_HOST", cfg.get("host"))
    port = int(os.getenv("REDIS_PORT", cfg.get("port", 6379)))
    client = redis.Redis(host=host, port=port, decode_responses=True)
    try:
        return RedisGraph(name, client)
    except Exception:
        return None


def get_s3_storage() -> S3Storage | None:
    """Return :class:`S3Storage` if bucket information is configured."""

    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        return None
    if boto3 is None:
        return None
    prefix = os.getenv("S3_PREFIX", "")
    aws_kwargs = {}
    if endpoint := os.getenv("S3_ENDPOINT_URL"):
        aws_kwargs["endpoint_url"] = endpoint
    if key := os.getenv("AWS_ACCESS_KEY_ID"):
        aws_kwargs["aws_access_key_id"] = key
    if secret := os.getenv("AWS_SECRET_ACCESS_KEY"):
        aws_kwargs["aws_secret_access_key"] = secret
    client = boto3.client("s3", **aws_kwargs) if aws_kwargs else boto3.client("s3")
    return S3Storage(bucket, prefix, client=client)
