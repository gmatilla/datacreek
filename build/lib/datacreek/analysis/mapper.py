import logging
from typing import Callable, Iterable, List, Set, Tuple

import networkx as nx

from .monitoring import update_metric


def mapper_cover(graph: nx.Graph, radius: int = 1) -> List[Set[object]]:
    """Return a greedy cover of ``graph`` using BFS balls of ``radius``.

    The function repeatedly selects an uncovered node and forms a ball of
    radius ``radius`` around it until all nodes are covered.
    """
    remaining = set(graph.nodes())
    cover: List[Set[object]] = []
    while remaining:
        seed = remaining.pop()
        ball = set(
            nx.single_source_shortest_path_length(graph, seed, cutoff=radius).keys()
        )
        cover.append(ball | {seed})
        remaining.difference_update(ball)
    return cover


def mapper_nerve(
    graph: nx.Graph, radius: int = 1
) -> Tuple[nx.Graph, List[Set[object]]]:
    """Return the Mapper nerve of ``graph`` and the covering used."""
    cover = mapper_cover(graph, radius)
    nerve = nx.Graph()
    nerve.add_nodes_from(range(len(cover)))
    for i, A in enumerate(cover):
        for j in range(i + 1, len(cover)):
            if A & cover[j]:
                nerve.add_edge(i, j)
    return nerve, cover


def inverse_mapper(nerve: nx.Graph, cover: Iterable[Set[object]]) -> nx.Graph:
    """Reconstruct a graph from its Mapper ``nerve`` and ``cover`` sets."""
    cover_list = [set(c) for c in cover]
    g = nx.Graph()
    for cluster in cover_list:
        for node in cluster:
            g.add_node(node)
        for u in cluster:
            for v in cluster:
                if u != v:
                    g.add_edge(u, v)
    for u, v in nerve.edges():
        for a in cover_list[int(u)]:
            for b in cover_list[int(v)]:
                if a != b:
                    g.add_edge(a, b)
    return g


def default_lens(graph: nx.Graph) -> dict[object, float]:
    """Return lens values for ``graph`` using GraphWave energy or PCA."""

    import numpy as np

    try:  # prefer GraphWave energy if heavy deps available
        from .fractal import graphwave_embedding_chebyshev

        emb = graphwave_embedding_chebyshev(graph, scales=[1.0], num_points=4, order=3)
        return {n: float(np.linalg.norm(v)) for n, v in emb.items()}
    except Exception:  # pragma: no cover - fallback
        from sklearn.decomposition import PCA

        arr = nx.to_numpy_array(graph)
        if arr.size == 0:
            return {}
        comp = PCA(n_components=1).fit_transform(arr)
        return {n: float(comp[i, 0]) for i, n in enumerate(graph.nodes())}


def mapper_full(
    graph: nx.Graph,
    *,
    lens: Callable[[nx.Graph], dict[object, float]] | None = None,
    cover: tuple[int, float] = (10, 0.5),
    clusterer: str = "dbscan",
) -> tuple[nx.Graph, list[set[object]]]:
    """Return configurable Mapper nerve and cover of ``graph``."""

    import numpy as np

    if lens is None:
        lens_vals = default_lens(graph)
    else:
        lens_vals = lens(graph)

    if not lens_vals:
        return nx.Graph(), []

    nodes = list(graph.nodes())
    vals = np.asarray([lens_vals[n] for n in nodes], dtype=float)
    n_intervals, overlap = cover
    if n_intervals <= 0:
        raise ValueError("n_intervals must be positive")
    overlap = min(max(overlap, 0.0), 0.9)
    v_min, v_max = float(vals.min()), float(vals.max())
    width = (v_max - v_min) / max(1e-9, n_intervals - (n_intervals - 1) * overlap)
    step = width * (1 - overlap)

    intervals = [
        (v_min + i * step, v_min + i * step + width) for i in range(n_intervals)
    ]
    cover_sets: list[set[object]] = []
    for start, end in intervals:
        bucket_nodes = [n for n in nodes if start <= lens_vals[n] <= end]
        if not bucket_nodes:
            continue
        subgraph = graph.subgraph(bucket_nodes)
        if clusterer == "dbscan":
            X = np.array([[lens_vals[n]] for n in bucket_nodes], dtype=float)
            try:
                from sklearn.cluster import DBSCAN

                labels = DBSCAN(eps=width * 0.5, min_samples=1).fit_predict(X)
            except Exception:  # pragma: no cover - sklearn missing
                labels = np.zeros(len(bucket_nodes), dtype=int)
            for lab in set(labels):
                cluster_nodes = {
                    bucket_nodes[i] for i, lb in enumerate(labels) if lb == lab
                }
                if cluster_nodes:
                    cover_sets.append(cluster_nodes)
        else:  # connected components
            for comp in nx.connected_components(subgraph):
                cover_sets.append(set(comp))

    nerve = nx.Graph()
    nerve.add_nodes_from(range(len(cover_sets)))
    for i, A in enumerate(cover_sets):
        for j in range(i + 1, len(cover_sets)):
            if A & cover_sets[j]:
                nerve.add_edge(i, j)
    return nerve, cover_sets


def mapper_to_json(
    graph: nx.Graph,
    *,
    lens: Callable[[nx.Graph], dict[object, float]] | None = None,
    cover: tuple[int, float] = (10, 0.5),
    clusterer: str = "dbscan",
    autotune: bool = True,
    overlaps: Iterable[float] = (0.2, 0.3, 0.4, 0.5),
) -> str:
    """Return Mapper nerve and cover exported as JSON for the ``/explain`` API.

    Parameters
    ----------
    graph:
        Input graph for which to compute the Mapper nerve.
    lens:
        Optional mapping from node to scalar lens value.
    cover:
        Tuple ``(n_intervals, overlap)`` controlling the interval cover. When
        ``autotune`` is ``True`` only ``n_intervals`` is used and the overlap is
        optimised over ``overlaps``.
    clusterer:
        Clustering strategy passed to :func:`mapper_full`.
    autotune:
        If ``True`` (default) perform overlap grid-search via :func:`tune_overlap`
        and expose the chosen value through the ``mapper_overlap_opt`` metric.
    overlaps:
        Candidate overlap ratios evaluated during autotuning.
    """

    import json

    n_intervals, overlap = cover
    if autotune:
        nerve, cover_sets, _ = tune_overlap(
            graph,
            overlaps=overlaps,
            n_intervals=n_intervals,
            lens=lens,
            clusterer=clusterer,
        )
    else:
        nerve, cover_sets = mapper_full(
            graph, lens=lens, cover=(n_intervals, overlap), clusterer=clusterer
        )
    data = {
        "nodes": [{"id": i, "members": list(c)} for i, c in enumerate(cover_sets)],
        "links": [{"source": u, "target": v} for u, v in nerve.edges()],
    }
    return json.dumps(data)


def _cover_silhouette(
    lens_vals: dict[object, float], cover_sets: Iterable[set[object]]
) -> float:
    """Return the silhouette coefficient of ``cover_sets`` w.r.t ``lens_vals``.

    Parameters
    ----------
    lens_vals:
        Mapping of node to lens value.
    cover_sets:
        Clusters produced by :func:`mapper_full`.

    Notes
    -----
    The silhouette coefficient compares mean intra-cluster distances with the
    smallest mean distance to points in other clusters. For the 1-D lens values
    used here this simplifies to differences along that axis. If scikit-learn
    is unavailable or less than two clusters exist, ``0.0`` is returned.
    """

    try:  # optional dependency, pragma: no cover for failure path
        from sklearn.metrics import silhouette_score  # type: ignore
    except Exception:  # pragma: no cover - sklearn missing
        silhouette_score = None  # type: ignore

    nodes = list(lens_vals.keys())
    labels = [-1] * len(nodes)
    for idx, cluster in enumerate(cover_sets):
        for n in cluster:
            i = nodes.index(n)
            if labels[i] == -1:
                labels[i] = idx  # assign node to first cluster encountered
    if len(set(labels)) <= 1 or -1 in labels:
        return 0.0

    X = [[lens_vals[n]] for n in nodes]
    if silhouette_score is not None:
        try:
            return float(silhouette_score(X, labels))
        except Exception:  # pragma: no cover - numerical issues
            return 0.0

    # Fallback: manual silhouette computation for 1-D lens values
    import numpy as np

    X = np.array(X, dtype=float).reshape(-1)
    labels_arr = np.array(labels)
    unique = np.unique(labels_arr)
    sils = []
    for i, x in enumerate(X):
        same = X[labels_arr == labels_arr[i]]
        other = [X[labels_arr == u] for u in unique if u != labels_arr[i]]
        a = float(np.mean(np.abs(same - x))) if len(same) > 1 else 0.0
        b = min(float(np.mean(np.abs(o - x))) for o in other) if other else 0.0
        sils.append((b - a) / max(a, b) if max(a, b) else 0.0)
    return float(np.mean(sils))


def autotune_mapper_overlap(
    graph: nx.Graph,
    *,
    overlaps: Iterable[float] = (0.2, 0.3, 0.4, 0.5),
    n_intervals: int = 10,
    lens: Callable[[nx.Graph], dict[object, float]] | None = None,
    clusterer: str = "dbscan",
) -> tuple[nx.Graph, list[set[object]], float, float]:
    """Grid-search the Mapper overlap maximizing the silhouette score.

    The function evaluates :func:`mapper_full` for each candidate overlap and
    selects the configuration whose clustering of nodes (cover sets) yields the
    highest silhouette coefficient. The silhouette coefficient ``s`` for an
    overlap ``\omega`` is computed as

    .. math::

        s(\omega) = \frac{b - a}{\max(a, b)}

    where ``a`` is the mean intra-cluster distance and ``b`` the minimum mean
    distance to points of other clusters. A higher value indicates better
    separated clusters.

    Returns
    -------
    nerve, cover, best_overlap, best_score
        The Mapper nerve and cover with the best overlap along with the
        optimal overlap value and corresponding silhouette coefficient.
    """

    if lens is None:
        lens_vals = default_lens(graph)
    else:
        lens_vals = lens(graph)
    if not lens_vals:
        return nx.Graph(), [], 0.5, 0.0

    best_score = -1.0
    best_result = (nx.Graph(), [], 0.5, 0.0)
    for ov in overlaps:
        nerve, cover_sets = mapper_full(
            graph,
            lens=lambda _g: lens_vals,
            cover=(n_intervals, ov),
            clusterer=clusterer,
        )
        score = _cover_silhouette(lens_vals, cover_sets)
        if score > best_score:
            best_score = score
            best_result = (nerve, cover_sets, ov, score)
    return best_result


def tune_overlap(
    graph: nx.Graph,
    overlaps: Iterable[float] = (0.2, 0.3, 0.4, 0.5),
    *,
    n_intervals: int = 10,
    lens: Callable[[nx.Graph], dict[object, float]] | None = None,
    clusterer: str = "dbscan",
) -> tuple[nx.Graph, list[set[object]], float]:
    """Return Mapper nerve and cover for the best overlap.

    Parameters
    ----------
    graph:
        Input graph on which to run the Mapper algorithm.
    overlaps:
        Candidate overlap ratios to evaluate.
    n_intervals:
        Number of intervals used in the cover for each overlap.
    lens:
        Optional lens function mapping nodes to scalar values.
    clusterer:
        Clustering strategy passed to :func:`mapper_full`.

    Returns
    -------
    nerve, cover, best_overlap
        The Mapper nerve and cover corresponding to the overlap that maximizes
        the silhouette score. The chosen overlap is also returned. The selected
        value is exposed via the Prometheus metric ``mapper_overlap_opt``.
    """

    nerve, cover, best_overlap, _ = autotune_mapper_overlap(
        graph,
        overlaps=overlaps,
        n_intervals=n_intervals,
        lens=lens,
        clusterer=clusterer,
    )
    update_metric("mapper_overlap_opt", float(best_overlap))
    return nerve, cover, best_overlap


import os
import pickle  # nosec B403
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

try:  # optional dependencies
    import redis
except Exception:  # pragma: no cover - optional
    redis = None  # type: ignore

try:
    import lmdb  # type: ignore
except Exception:  # pragma: no cover - optional
    lmdb = None  # type: ignore

from ..utils.cache import cache_l1
from ..utils.config import load_config
from ..utils.evict_log import log_eviction

cfg = load_config()
cache_cfg = cfg.get("cache", {})
_redis_ttl = int(cache_cfg.get("l1_ttl_init", 3600))
_ttl_min = int(cache_cfg.get("l1_ttl_min", 300))
_ttl_max = int(cache_cfg.get("l1_ttl_max", 7200))

_evict_thread: threading.Thread | None = None
_evict_stop = threading.Event()

# Adaptive TTL parameters for Redis L1 cache
_redis_hits = 0
_redis_misses = 0
_last_ttl_eval = time.time()
_hit_ema = 0.0
_ALPHA = 0.3


def _adjust_ttl(client: Optional["redis.Redis"], key: str | None = None) -> None:
    """Adjust Redis TTL based on hit ratio and CPU load every 5 minutes."""

    global _redis_hits, _redis_misses, _redis_ttl, _last_ttl_eval, _hit_ema
    now = time.time()
    if now - _last_ttl_eval < 300:
        return
    total = _redis_hits + _redis_misses
    ratio = _redis_hits / max(1, total)
    _hit_ema = _ALPHA * ratio + (1 - _ALPHA) * _hit_ema
    try:
        from .monitoring import redis_hit_ratio as _hit_gauge

        if _hit_gauge is not None:
            _hit_gauge.set(_hit_ema)
    except Exception:  # pragma: no cover - optional metrics
        pass
    if client is not None:
        try:
            client.config_set("maxmemory-policy", "allkeys-lru")
        except Exception:
            pass
    prev_ttl = _redis_ttl
    if _hit_ema < 0.2:
        _redis_ttl = max(int(_redis_ttl * 0.5), _ttl_min)
    elif _hit_ema > 0.8:
        _redis_ttl = min(int(_redis_ttl * 1.2), _ttl_max)
    try:
        load = os.getloadavg()[0] / max(1, os.cpu_count() or 1)
        if load > 0.7:
            _redis_ttl = max(int(_redis_ttl * 1.5), _ttl_max)
    except Exception:
        pass
    if _redis_ttl != prev_ttl:
        logging.getLogger(__name__).debug("L1 TTL updated to %d", _redis_ttl)
    if key and client is not None:
        try:
            client.expire(key, _redis_ttl)
        except Exception:
            pass
    _redis_hits = 0
    _redis_misses = 0
    _last_ttl_eval = now


def _cache_put(
    key: str,
    nerve: nx.Graph,
    cover: Iterable[set[object]],
    *,
    redis_client: Optional["redis.Redis"] = None,
    lmdb_path: str = "lmdb/hot_graph.mdb",
    ssd_dir: str = "nerve_cache",
    ttl: int | None = None,
) -> None:
    """Store ``nerve`` and ``cover`` in hierarchical caches.

    Parameters
    ----------
    key:
        Identifier for the cached subgraph.
    nerve, cover:
        Mapper nerve representation to persist.
    ttl:
        Time-to-live in seconds for the Redis entry. Defaults to one hour.
    """
    data = pickle.dumps((nx.node_link_data(nerve), [list(c) for c in cover]))
    ttl_val = int(ttl if ttl is not None else _redis_ttl)
    if redis_client is None and redis is not None:  # pragma: no cover - fallback
        try:
            redis_client = redis.Redis()
        except Exception:
            redis_client = None
    if redis_client is not None:
        try:
            redis_client.setex(key, ttl_val, data)
            redis_client.incr("miss")
        except Exception:  # pragma: no cover - network errors
            pass
    if lmdb is not None:
        try:
            from ..utils.config import load_config

            cfg = load_config()
            cache_cfg = cfg.get("cache", {})
            limit_mb = int(cache_cfg.get("l2_max_size_mb", 256))
            ttl_h = int(cache_cfg.get("l2_ttl_hours", 24))
            start_l2_eviction_thread(lmdb_path)
            env = lmdb.open(lmdb_path, map_size=limit_mb << 20)
            env.set_mapsize(limit_mb * 1024**2)
            stat = env.stat()
            info = env.info()
            size_mb = (stat["psize"] * info["map_size"]) / (1 << 20)
            if size_mb > 0.9 * limit_mb:
                logging.getLogger(__name__).warning(
                    "LMDB soft-quota reached %.1f MB", size_mb
                )
                env.close()
                return
            if size_mb > limit_mb:  # pragma: no cover - requires large LMDB
                _l2_evict_once(env, limit_mb, ttl_h)
            with env.begin(write=True) as txn:
                now = time.time()
                txn.put(key.encode(), pickle.dumps((now, data)))
                stat = env.stat()
                info = env.info()
                current_mb = (stat["psize"] * info["map_size"]) / (1 << 20)
                cur = txn.cursor()
                n_deleted = 0
                prev_mb = current_mb
                if cur.first():  # pragma: no cover - deterministic pruning
                    while True:
                        raw = cur.value()
                        ts, _ = pickle.loads(raw)  # nosec B301
                        age_h = (now - ts) / 3600
                        if age_h > ttl_h or current_mb > limit_mb:
                            cur.delete()
                            n_deleted += 1
                        if not cur.next():
                            break
                        stat = env.stat()
                        info = env.info()
                        current_mb = (stat["psize"] * info["map_size"]) / (1 << 20)
                    env.sync()  # pragma: no cover - I/O heavy
                    if n_deleted:
                        logging.getLogger(__name__).info(
                            "[L2-EVICT] %d keys purged (%.1f MB→%.1f MB)",
                            n_deleted,
                            prev_mb,
                            current_mb,
                        )
                env.sync()  # pragma: no cover - I/O heavy
            env.close()
        except Exception:  # pragma: no cover - disk errors
            pass
    try:
        Path(ssd_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(ssd_dir) / f"{key}.pkl", "wb") as fh:
            fh.write(data)
    except Exception:  # pragma: no cover - disk errors
        pass


@cache_l1
def _cache_get(
    key: str,
    *,
    redis_client: Optional["redis.Redis"] = None,
    lmdb_path: str = "lmdb/hot_graph.mdb",
    ssd_dir: str = "nerve_cache",
) -> Optional[tuple[nx.Graph, list[set[object]]]]:
    """Retrieve cached Mapper nerve from hierarchical caches."""
    global _redis_hits, _redis_misses
    blob = None
    if redis_client is None and redis is not None:  # pragma: no cover
        try:
            redis_client = redis.Redis()
        except Exception:
            redis_client = None
    if redis_client is not None:
        try:
            blob = redis_client.get(key)
        except Exception:
            blob = None
    if blob is not None:
        _redis_hits += 1
        if redis_client is not None:
            try:
                redis_client.incr("hits")
            except Exception:
                pass
    else:
        _redis_misses += 1
    _adjust_ttl(redis_client, key)
    if blob is None and lmdb is not None:
        try:
            from ..utils.config import load_config

            cfg = load_config()
            ttl_h = int(cfg.get("cache", {}).get("l2_ttl_hours", 24))
            env = lmdb.open(lmdb_path, readonly=True, lock=False)
            with env.begin() as txn:
                blob = txn.get(key.encode())
            env.close()
            if blob is not None:
                ts, blob = pickle.loads(blob)  # nosec B301
                if (time.time() - ts) / 3600 > ttl_h:
                    blob = None
        except Exception:
            blob = None
    if blob is None:
        try:
            with open(Path(ssd_dir) / f"{key}.pkl", "rb") as fh:
                blob = fh.read()
        except Exception:
            blob = None
    if blob is None:
        return None
    nerve_data, cover_data = pickle.loads(blob)  # nosec B301
    nerve = nx.node_link_graph(nerve_data)
    cover = [set(c) for c in cover_data]
    return nerve, cover


def _hash_graph(graph: nx.Graph) -> str:
    """Return SHA1 hash of ``graph`` structure."""

    import hashlib

    edges = sorted((sorted((str(u), str(v))) for u, v in graph.edges()))
    nodes = sorted(str(n) for n in graph.nodes())
    h = hashlib.sha1()  # nosec B324
    h.update(pickle.dumps((nodes, edges)))
    return h.hexdigest()


def cache_mapper_nerve(
    graph: nx.Graph,
    radius: int,
    *,
    redis_client: Optional["redis.Redis"] = None,
    lmdb_path: str = "hot_subgraph",
    ssd_dir: str = "nerve_cache",
    ttl: int = 3600,
) -> tuple[nx.Graph, list[set[object]]]:
    """Return cached Mapper nerve for ``graph`` or compute it."""

    key = f"{radius}_{_hash_graph(graph)}"
    res = _cache_get(
        key, redis_client=redis_client, lmdb_path=lmdb_path, ssd_dir=ssd_dir
    )
    if res is not None:
        return res

    nerve, cover = mapper_nerve(graph, radius)
    _cache_put(
        key,
        nerve,
        cover,
        redis_client=redis_client,
        lmdb_path=lmdb_path,
        ssd_dir=ssd_dir,
        ttl=None,
    )
    return nerve, cover


def _l2_evict_once(env, limit_mb: int, ttl_h: int) -> None:
    """Purge expired or oversized LMDB entries from ``env``."""

    now = time.time()
    with env.begin(write=True) as txn:
        stat = env.stat()
        info = env.info()
        size_mb = (stat["psize"] * info["map_size"]) / (1 << 20)
        soft = info["map_size"] > limit_mb * 1024**2 * 0.9
        cur = txn.cursor()
        n_deleted = 0
        prev_mb = size_mb
        if cur.first():
            while True:
                raw = cur.value()
                ts, _ = pickle.loads(raw)  # nosec B301
                age_h = (now - ts) / 3600
                if age_h > ttl_h or size_mb > limit_mb:
                    key_bytes = cur.key()
                    cur.delete()
                    n_deleted += 1
                    cause = "ttl" if age_h > ttl_h else "quota"
                    log_eviction(key_bytes.decode(errors="ignore"), now, cause)
                    if soft:
                        logging.getLogger(__name__).debug(
                            "LMDB-EVICT %s", key_bytes.decode(errors="ignore")
                        )
                    try:
                        from ..analysis.monitoring import redis_evictions_l2_total

                        if redis_evictions_l2_total is not None:
                            redis_evictions_l2_total.inc()
                    except Exception:
                        pass
                if not cur.next():
                    break
                stat = env.stat()
                info = env.info()
                size_mb = (stat["psize"] * info["map_size"]) / (1 << 20)
        if n_deleted:
            logging.getLogger(__name__).info(
                "[L2-EVICT] %d keys purged (%.1f MB→%.1f MB)",
                n_deleted,
                prev_mb,
                size_mb,
            )
        env.sync()


def delete_l2_entry(env, key: str) -> bool:
    """Delete ``key`` from ``env`` and log a manual eviction."""

    with env.begin(write=True) as txn:
        existed = txn.delete(key.encode())
    if existed:
        log_eviction(key, time.time(), "manual")
    return bool(existed)


def _evict_worker(path: str, limit_mb: int, ttl_h: int, interval: float) -> None:
    """Background worker periodically evicting LMDB entries."""

    if lmdb is None:
        return
    while not _evict_stop.wait(interval):
        try:
            env = lmdb.open(path, map_size=limit_mb << 20)
            env.set_mapsize(limit_mb * 1024**2)
            stat = env.stat()
            info = env.info()
            size_mb = (stat["psize"] * info["map_size"]) / (1 << 20)
            if size_mb > limit_mb:
                _l2_evict_once(env, limit_mb, ttl_h)
            env.close()
        except Exception:  # pragma: no cover - disk errors
            pass


def start_l2_eviction_thread(
    lmdb_path: str = "lmdb/hot_graph.mdb", *, interval: float = 3600.0
) -> None:
    """Start daemon thread cleaning LMDB cache according to TTL."""

    global _evict_thread
    if _evict_thread is not None:
        return
    from ..utils.config import load_config

    cfg = load_config()
    cache_cfg = cfg.get("cache", {})
    limit_mb = int(cache_cfg.get("l2_max_size_mb", 2048))
    ttl_h = int(cache_cfg.get("l2_ttl_hours", 24))
    _evict_stop.clear()
    t = threading.Thread(
        target=_evict_worker,
        args=(lmdb_path, limit_mb, ttl_h, interval),
        daemon=True,
    )
    t.start()
    _evict_thread = t


def stop_l2_eviction_thread() -> None:
    """Stop the LMDB eviction worker if running."""

    global _evict_thread
    if _evict_thread is None:
        return
    _evict_stop.set()
    _evict_thread.join(timeout=0.5)
    _evict_thread = None
