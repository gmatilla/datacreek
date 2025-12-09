import importlib
import importlib.abc
import importlib.util
import os
import sys
import time
import types
from pathlib import Path

import pytest

pytest.importorskip("networkx")
pytest.importorskip("numpy")
import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

spec = importlib.util.spec_from_file_location(
    "datacreek.plugins.pgvector_export",
    Path(__file__).resolve().parents[1]
    / "datacreek"
    / "plugins"
    / "pgvector_export.py",
)
pgvector_export = importlib.util.module_from_spec(spec)
assert isinstance(spec.loader, importlib.abc.Loader)
spec.loader.exec_module(pgvector_export)


class DummyKG:
    def __init__(self):
        self.graph = nx.Graph()


class DummyCopy:
    def __init__(self, sql):
        self.sql = sql
        self.rows = []

    def write_row(self, row):
        self.rows.append(row)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class DummyCursor:
    def __init__(self):
        self.calls = []
        self.copies = []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))

    def copy(self, sql):
        cp = DummyCopy(sql)
        self.copies.append(cp)
        return cp

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class DummyConn:
    def __init__(self):
        self.cur = DummyCursor()
        self.committed = False

    def cursor(self):
        return self.cur

    def commit(self):
        self.committed = True


def test_export_embeddings_latency():
    kg = DummyKG()
    kg.graph.add_node("a", embedding=[0.1, 0.2], poincare_embedding=[0.3, 0.4])
    kg.graph.add_node("b", embedding=[0.5, 0.6], poincare_embedding=[0.7, 0.8])
    kg.graph.graph["faiss_meta"] = {"embedding_version": "1.0"}
    conn = DummyConn()

    start = time.perf_counter()
    rows = pgvector_export.export_embeddings_pg(
        kg, conn, table="emb", lists=10, embedding_version="1.0"
    )
    duration = time.perf_counter() - start

    assert rows == 4
    assert conn.committed
    assert duration < 0.03
    assert any("embedding_version" in cp.sql for cp in conn.cur.copies)
    assert any("ivfflat" in sql for sql, _ in conn.cur.calls)


def test_query_topk_sql():
    conn = DummyConn()
    vec = [0.1, 0.2]
    pgvector_export.query_topk_pg(conn, table="emb", vec=vec, k=3, version="1.0")
    assert any("embedding_version" in sql for sql, _ in conn.cur.calls)


def test_pgvector_query_metric(monkeypatch):
    conn = DummyConn()
    recorded = {}

    def fake_update(name, value, labels=None):
        recorded[name] = value

    fake_mod = types.SimpleNamespace(update_metric=fake_update)
    monkeypatch.setitem(sys.modules, "datacreek.analysis.monitoring", fake_mod)
    vec = [0.1, 0.2]
    pgvector_export.query_topk_pg(conn, table="emb", vec=vec, k=2, version="1.0")
    assert "pgvector_query_ms" in recorded
    assert recorded["pgvector_query_ms"] >= 0


@pytest.mark.faiss_gpu
def test_recall_vs_faiss():
    faiss = pytest.importorskip("faiss")
    rng = np.random.default_rng(0)
    xb = rng.standard_normal((200, 4)).astype("float32")
    xq = rng.standard_normal((10, 4)).astype("float32")
    index = faiss.IndexFlatIP(4)
    index.add(xb)
    _, gt = index.search(xq, 5)
    results = []
    for q in xq:
        sims = xb @ q
        ids = np.argsort(sims)[::-1][:5]
        results.append(ids)
    recall = sum(len(set(r).intersection(g)) for r, g in zip(results, gt)) / (
        len(xq) * 5
    )
    assert recall == 1.0


@pytest.mark.heavy
def test_pgvector_latency_recall(tmp_path):
    dsn = os.environ.get("PGVECTOR_URL")
    if not dsn:
        pytest.skip("pgvector not configured")
    psycopg = pytest.importorskip("psycopg")
    faiss = pytest.importorskip("faiss")

    rng = np.random.default_rng(0)
    dim = 8
    if os.environ.get("PGVECTOR_FULL_BENCH"):
        n = 1_000_000
        q = 1000
    else:
        n = 1000
        q = 100
    xb = rng.standard_normal((n, dim)).astype("float32")
    xq = rng.standard_normal((q, dim)).astype("float32")

    # baseline recall using FAISS CPU exact search
    index = faiss.IndexFlatIP(dim)
    index.add(xb)
    _, gt = index.search(xq, 5)

    # build graph structure to reuse export helper
    kg = DummyKG()
    for i, vec in enumerate(xb):
        kg.graph.add_node(str(i), embedding=vec)

    with psycopg.connect(dsn) as conn:
        conn.execute("DROP TABLE IF EXISTS emb")
        # store vectors in dedicated table for latency benchmark
        pgvector_export.export_embeddings_pg(
            kg, conn, table="emb", lists=100, embedding_version="1.0"
        )
        t0 = time.perf_counter()
        rows = [
            pgvector_export.query_topk_pg(conn, "emb", q, k=5, version="1.0")
            for q in xq
        ]
        elapsed = (time.perf_counter() - t0) / len(xq)

    recall = 0
    for r, g in zip(rows, gt):
        ids = [int(i[0]) for i in r]
        recall += len(set(ids).intersection(g))
    recall /= len(xq) * 5

    assert elapsed * 1000 < 30
    assert recall >= 0.9
