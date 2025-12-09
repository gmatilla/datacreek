# coverage: ignore-file
import hashlib
import logging
import math
import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

try:  # optional dependency
    import gudhi as gd
    import gudhi.representations as gr
except Exception:  # pragma: no cover - optional dependency missing
    gd = None  # type: ignore
    gr = None  # type: ignore
import concurrent.futures
import time

import networkx as nx
import numpy as np


def with_timeout(
    timeout: float,
    *,
    counter: Optional[Any] = None,
    duration_gauge: Optional[Any] = None,
    fallback: Optional[Callable[..., float]] = None,
) -> Callable[[Callable[..., float]], Callable[..., float]]:
    """Return a decorator executing ``func`` with ``timeout`` seconds.

    On timeout, ``counter`` is incremented and ``fallback`` is executed.
    The execution duration is stored in ``duration_gauge`` when provided.
    """

    def decorator(func: Callable[..., float]) -> Callable[..., float]:
        def wrapper(*args, **kwargs) -> float:
            start = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(func, *args, **kwargs)
                try:
                    result = future.result(timeout=timeout)
                    duration = time.monotonic() - start
                    if duration_gauge is not None:
                        try:
                            duration_gauge.set(duration)
                        except Exception:
                            pass
                    return float(result)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    if counter is not None:
                        try:
                            counter.inc()
                        except Exception:
                            pass
                    if duration_gauge is not None:
                        try:
                            duration_gauge.set(timeout)
                        except Exception:
                            pass
                    if fallback is not None:
                        return float(fallback(*args, **kwargs))
                    raise

        return wrapper

    return decorator


try:  # optional
    from neo4j import Driver
except Exception:  # pragma: no cover - optional dependency missing
    from typing import Any

    Driver = Any  # type: ignore

try:
    from scipy.linalg import eigh  # type: ignore
    from scipy.sparse import csgraph  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    from numpy.linalg import eigh  # type: ignore

    csgraph = None  # type: ignore


def ensure_graphrnn_checkpoint(
    cfg: Mapping[str, Any] | None = None,
    cache_dir: str | os.PathLike | None = None,
) -> Path | None:  # pragma: no cover - uses network and heavy deps
    """Download GraphRNN checkpoint from S3 if configured.

    Parameters
    ----------
    cfg:
        Configuration mapping. ``load_config()`` is used when ``None``.
    cache_dir:
        Directory in which to save the checkpoint. Defaults to
        ``DATACREEK_CACHE``.

    Returns
    -------
    Path | None
        Local path to the checkpoint or ``None`` on failure.
    """

    from ..utils.config import load_config

    cfg = load_config() if cfg is None else cfg
    tpl_cfg = cfg.get("tpl", {})
    bucket = tpl_cfg.get("rnn_ckpt_bucket")
    key = tpl_cfg.get("rnn_ckpt_key")
    sha = tpl_cfg.get("rnn_ckpt_sha")
    if not bucket or not key or not sha:
        return None

    path = Path(cache_dir or os.getenv("DATACREEK_CACHE", "./cache"))
    path.mkdir(parents=True, exist_ok=True)
    local = path / Path(key).name
    if local.exists():
        with open(local, "rb") as fh:
            if hashlib.sha256(fh.read()).hexdigest() == sha:
                return local

    try:
        import boto3

        s3 = boto3.client("s3")
        s3.download_file(bucket, key, str(local), ExtraArgs={"ChecksumMode": "ENABLED"})
    except Exception:  # pragma: no cover - network or dependency issues
        return None

    with open(local, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    if digest != sha:
        raise RuntimeError("graphrnn checkpoint sha mismatch")
    return local


def _laplacian(
    graph: nx.Graph, *, normed: bool = False
) -> np.ndarray:  # pragma: no cover - deterministic
    """Return Laplacian matrix of ``graph``."""

    if csgraph is not None:
        a = nx.to_scipy_sparse_array(graph)
        return csgraph.laplacian(a, normed=normed).toarray()
    a = nx.to_numpy_array(graph)
    # explicit numpy call avoids older numpy's ``_sum`` helper errors
    deg = np.sum(a, axis=1)
    lap = np.diag(deg) - a
    if normed:
        with np.errstate(divide="ignore"):
            inv_sqrt = 1.0 / np.sqrt(deg)
        inv_sqrt[~np.isfinite(inv_sqrt)] = 0.0
        lap = (inv_sqrt[:, None] * lap) * inv_sqrt
    return lap


def lanczos_lmax(L: np.ndarray, iters: int = 10) -> float:  # pragma: no cover - heavy
    """Return largest eigenvalue estimate using power iteration.

    Parameters
    ----------
    L:
        Symmetric positive semidefinite matrix.
    iters:
        Number of Lanczos iterations. More iterations yield a more accurate
        estimate at the cost of additional matrix-vector multiplications.
    """
    import numpy as np

    n = L.shape[0]
    if n <= 1000:
        try:
            import numpy.linalg as nla
            import scipy.sparse as sp

            return float(nla.eigvalsh(L.toarray() if sp.issparse(L) else L).max())
        except Exception:
            pass

    q = np.random.rand(n)
    q /= np.linalg.norm(q)
    for _ in range(iters):
        z = L.dot(q)
        q = z / (np.linalg.norm(z) + 1e-12)
    return float(q.dot(L.dot(q)))


def lanczos_top_eigenvalue(
    L: np.ndarray, k: int = 5
) -> float:  # pragma: no cover - heavy
    """Return largest eigenvalue via Lanczos with ``k`` iterations.

    Parameters
    ----------
    L:
        Symmetric matrix representing the operator.
    k:
        Number of Lanczos iterations.

    Notes
    -----
    Orthogonalization is performed at each step. The Rayleigh quotient of
    the final vector approximates the spectral radius.
    """

    import numpy as np

    n = L.shape[0]
    q = np.random.randn(n)
    q /= np.linalg.norm(q)
    basis = [q]
    for _ in range(k):
        z = L.dot(q)
        for b in basis:
            z -= b.dot(z) * b
        nrm = np.linalg.norm(z)
        if nrm == 0:
            break
        q = z / nrm
        basis.append(q)
    v = basis[-1]
    denom = v.dot(v)
    if denom == 0:
        return 0.0
    return float(v.dot(L.dot(v)) / denom)


def eigsh_safe(L: np.ndarray) -> float:  # pragma: no cover - heavy dependency
    """Return the dominant eigenvalue with a watchdog timeout.

    The timeout duration is configurable via ``spectral.eig_timeout``. When the
    underlying ``eigsh`` call exceeds this limit, the
    ``eigsh_timeouts_total`` counter is incremented and a Lanczos fallback is
    used instead.
    """

    from scipy.sparse.linalg import eigsh

    from datacreek.utils.config import load_config

    from .monitoring import eigsh_last_duration, eigsh_timeouts_total

    cfg = load_config()
    timeout_s = float(cfg.get("spectral", {}).get("eig_timeout", 60))

    def _run() -> float:
        return float(
            eigsh(
                L,
                k=1,
                which="LM",
                tol=1e-3,
                return_eigenvectors=False,
            )[0]
        )

    wrapped = with_timeout(
        timeout_s,
        counter=eigsh_timeouts_total,
        duration_gauge=eigsh_last_duration,
        fallback=lambda *_a, **_k: lanczos_top_eigenvalue(L, k=5),
    )(_run)

    return wrapped()


def eigsh_lmax_watchdog(
    L: np.ndarray, maxiter: int, timeout: float = 60.0
) -> float:  # pragma: no cover - heavy dependency
    """Return ``eigsh`` largest eigenvalue with timeout and fallback."""

    import numpy.linalg as nla
    import scipy.sparse as sp
    from scipy.sparse.linalg import ArpackNoConvergence, eigsh

    from .monitoring import eigsh_last_duration, eigsh_timeouts_total

    def _run() -> float:
        return float(
            eigsh(
                L,
                k=1,
                which="LM",
                tol=1e-3,
                maxiter=maxiter,
                return_eigenvectors=False,
            )[0]
        )

    wrapped = with_timeout(
        timeout,
        counter=eigsh_timeouts_total,
        duration_gauge=eigsh_last_duration,
        fallback=lambda *_a, **_k: lanczos_top_eigenvalue(L, k=5),
    )(_run)

    try:
        return wrapped()
    except ArpackNoConvergence:
        if eigsh_timeouts_total is not None:
            try:
                eigsh_timeouts_total.inc()
            except Exception:
                pass
        try:
            return lanczos_top_eigenvalue(L, k=5)
        except Exception:
            return float(nla.eigvalsh(L.toarray() if sp.issparse(L) else L).max())


def box_cover(graph: nx.Graph, radius: int) -> List[set[int]]:
    """Return a box covering of ``graph`` with radius ``radius``.

    Parameters
    ----------
    graph:
        Input graph.
    radius:
        Radius ``l_B`` used for the covering.

    Returns
    -------
    list[set[int]]
        List of node sets, each representing a box of diameter ``2 * radius``.
    """

    remaining = set(graph.nodes())
    boxes: List[set[int]] = []
    while remaining:
        start = remaining.pop()
        seen = nx.single_source_shortest_path_length(graph, start, cutoff=radius)
        box = {n for n in seen.keys() if n in remaining or n == start}
        remaining.difference_update(box)
        boxes.append(box)
    return boxes


def box_counting_dimension(
    graph: nx.Graph, radii: Iterable[int]
) -> Tuple[float, List[Tuple[int, int]]]:
    """Estimate fractal dimension of ``graph`` using box covering.

    Parameters
    ----------
    graph:
        Input graph.
    radii:
        Iterable of box radii ``l_B``. Larger radii correspond to coarser covers.

    Returns
    -------
    float
        Estimated fractal dimension based on the slope of ``log(N_B)`` vs ``log(1/l_B)``.
    List[Tuple[int, int]]
        List of ``(radius, box_count)`` pairs used for the estimation.
    """

    counts: List[Tuple[int, int]] = []
    nodes = list(graph.nodes())
    for r in radii:
        remaining = set(nodes)
        boxes = 0
        while remaining:
            start = remaining.pop()
            seen = nx.single_source_shortest_path_length(graph, start, cutoff=r)
            remaining.difference_update(seen.keys())
            boxes += 1
        counts.append((r, boxes))

    xs = [-math.log(float(r)) for r, _ in counts]
    ys = [math.log(float(n)) for _, n in counts]
    if len(xs) < 2:
        return 0.0, counts
    slope, _ = np.polyfit(xs, ys, 1)
    return slope, counts


def colour_box_dimension(
    graph: nx.Graph, radii: Iterable[int]
) -> tuple[float, list[tuple[int, int]]]:
    """Estimate fractal dimension using a simple COLOUR-box scheme.

    This approximates the GPU COLOUR-box algorithm by assigning boxes in
    parallel via colour classes. Each radius ``l_B`` is processed by
    greedily colouring nodes so that no two neighbouring nodes share a colour
    within that radius.

    Parameters
    ----------
    graph:
        Input graph.
    radii:
        Iterable of box radii ``l_B``.

    Returns
    -------
    tuple[float, list[tuple[int, int]]]
        Estimated dimension and the ``(radius, box_count)`` pairs.
    """

    counts: list[tuple[int, int]] = []
    for r in radii:
        color = {}
        current = 0
        for node in graph:
            if node in color:
                continue
            color[node] = current
            for nbr in nx.single_source_shortest_path_length(graph, node, cutoff=r):
                if nbr != node:
                    color[nbr] = current
            current += 1
        counts.append((r, current))

    xs = [-math.log(float(r)) for r, _ in counts]
    ys = [math.log(float(n)) for _, n in counts]
    if len(xs) < 2:
        return 0.0, counts
    slope, _ = np.polyfit(xs, ys, 1)
    return slope, counts


def persistence_entropy(
    graph: nx.Graph, dimension: int = 0
) -> float:  # pragma: no cover - requires gudhi and heavy computation
    """Return the persistence entropy for ``graph`` in a given dimension.

    Parameters
    ----------
    graph:
        Input graph.
    dimension:
        Homology dimension for which to compute the persistence entropy.

    Returns
    -------
    float
        Persistence entropy of the diagram in ``dimension``. Returns ``0.0`` if
        no finite intervals are present.
    """

    if gd is None or gr is None:
        raise RuntimeError("gudhi is required for persistence calculations")

    if gd is None:
        raise RuntimeError("gudhi is required for persistence calculations")

    st = gd.SimplexTree()
    for node in graph.nodes():
        st.insert([node], filtration=0.0)
    for u, v in graph.edges():
        st.insert([u, v], filtration=1.0)

    st.compute_persistence(persistence_dim_max=True)
    diag = np.array(st.persistence_intervals_in_dimension(dimension))
    if diag.size == 0:
        return 0.0
    diag = diag[np.isfinite(diag[:, 1])]
    if diag.size == 0:
        return 0.0
    entropy = gr.Entropy()
    return float(entropy.fit_transform([diag])[0])


def _compute_persistence_diagrams(
    graph: nx.Graph, max_dim: int
) -> Dict[int, np.ndarray]:
    """Return persistence diagrams for ``graph`` using a clique filtration."""

    st = gd.SimplexTree()
    for node in graph.nodes():
        st.insert([node], filtration=0.0)

    # Insert all cliques up to size ``max_dim + 1`` to build the clique complex
    for clique in nx.enumerate_all_cliques(graph):
        dim = len(clique) - 1
        if dim == 0 or dim > max_dim:
            continue
        st.insert(clique, filtration=float(dim))

    st.compute_persistence(persistence_dim_max=True)
    diags: Dict[int, np.ndarray] = {}
    for dim in range(max_dim + 1):
        diag = np.array(st.persistence_intervals_in_dimension(dim))
        diag = diag[np.isfinite(diag[:, 1])]
        diags[dim] = diag
    return diags


@lru_cache(maxsize=128)
def _persistence_diagrams_cached(
    nodes: Tuple[int, ...], edges: Tuple[Tuple[int, int], ...], max_dim: int
) -> Dict[int, np.ndarray]:
    """LRU-cached helper keyed by nodes and edges."""

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return _compute_persistence_diagrams(g, max_dim)


def persistence_diagrams(
    graph: nx.Graph, max_dim: int = 2
) -> Dict[int, np.ndarray]:  # pragma: no cover - requires gudhi and heavy computation
    """Return persistence diagrams of ``graph`` up to ``max_dim`` using the clique complex.

    Results are cached based on the hash of nodes and edges to avoid expensive
    recomputation on repeated calls.
    """

    if gd is None:
        raise RuntimeError("gudhi is required for persistence calculations")

    nodes = tuple(sorted(graph.nodes()))
    edges = tuple(sorted(tuple(sorted((u, v))) for u, v in graph.edges()))
    return _persistence_diagrams_cached(nodes, edges, max_dim)


def graphwave_embedding(
    graph: nx.Graph, scales: Iterable[float], num_points: int = 10
) -> Dict[int, np.ndarray]:  # pragma: no cover - heavy spectral computation
    """Return GraphWave embeddings for ``graph``.

    Parameters
    ----------
    graph:
        Input graph.
    scales:
        Iterable of diffusion scales used for the wavelets.
    num_points:
        Number of sample points for the characteristic function.

    Returns
    -------
    dict
        Mapping of node to embedding vector of length ``len(scales) * num_points * 2``.
    """

    nodes = list(graph.nodes())
    a = nx.to_numpy_array(graph, nodelist=nodes)
    lap = _laplacian(graph, normed=False)
    evals, evecs = eigh(lap)
    ts = np.linspace(0, 2 * np.pi, num_points)
    emb: Dict[int, List[float]] = {n: [] for n in nodes}

    for s in scales:
        heat = evecs @ np.diag(np.exp(-s * evals)) @ evecs.T
        for idx, node in enumerate(nodes):
            coeffs = np.exp(1j * np.outer(ts, heat[idx, :])).sum(axis=1)
            emb[node].extend(coeffs.real)
            emb[node].extend(coeffs.imag)

    return {n: np.asarray(v, dtype=float) for n, v in emb.items()}


def chebyshev_heat_kernel(
    L: np.ndarray, t: float, m: int = 7
) -> np.ndarray:  # pragma: no cover - heavy spectral computation
    """Return Chebyshev approximation of ``exp(-t L)`` using 7 terms.

    Parameters
    ----------
    L:
        Laplacian matrix.
    t:
        Diffusion scale.
    m:
        Order of the approximation.

    Returns
    -------
    np.ndarray
        Approximated heat kernel matrix.
    """

    import numpy as np
    import numpy.linalg as nla
    import scipy.sparse as sp
    from scipy.special import iv

    from datacreek.utils.config import load_config

    cfg = load_config()
    eig_maxit = int(cfg.get("spectral", {}).get("eig_maxit", 2000))

    try:  # pragma: no cover - prefer sparse eigs when available
        if L.shape[0] > 2_000_000:
            lmax = lanczos_lmax(L, iters=5)
        else:
            lmax = eigsh_lmax_watchdog(L, eig_maxit, timeout=60.0)
    except Exception:  # fallback to dense eig
        import scipy.sparse as sp

        lmax = float(nla.eigvalsh(L.toarray() if sp.issparse(L) else L).max())

    if lmax == 0.0:
        lmax = 1.0

    n = L.shape[0]
    L_norm = (2.0 / lmax) * L - sp.eye(n, format="csr")
    ak = [2 * iv(k, t * lmax / 2.0) for k in range(m + 1)]
    a0, a1 = ak[0], ak[1]

    Tk_minus = sp.eye(n, format="csr")
    Tk = L_norm
    psi = a0 * Tk_minus + a1 * Tk
    for k in range(2, m + 1):
        Tk_plus = 2 * L_norm.dot(Tk) - Tk_minus
        psi = psi + ak[k] * Tk_plus
        Tk_minus, Tk = Tk, Tk_plus

    return np.exp(-t * lmax / 2.0) * psi


def graphwave_embedding_chebyshev(
    graph: nx.Graph,
    scales: Iterable[float],
    *,
    num_points: int = 10,
    order: int = 7,
) -> Dict[int, np.ndarray]:  # pragma: no cover - heavy spectral computation
    """Return GraphWave embeddings using Chebyshev approximation.

    The heat kernel :math:`e^{-sL}` is expanded in Chebyshev polynomials of the
    scaled Laplacian. This avoids the expensive full eigen-decomposition and
    scales linearly with ``order`` and ``|E|``.

    Parameters
    ----------
    graph:
        Input graph.
    scales:
        Diffusion scales used for the wavelets.
    num_points:
        Number of sample points for the characteristic function.
    order:
        Degree ``m`` of the Chebyshev approximation. ``m=7`` gives a good
        trade-off between accuracy and speed.

    Returns
    -------
    dict
        Mapping of node to embedding vectors.
    """

    nodes = list(graph.nodes())
    lap = _laplacian(graph, normed=False)
    ts = np.linspace(0, 2 * np.pi, num_points)
    emb: Dict[int, List[float]] = {n: [] for n in nodes}

    for s in scales:
        heat = chebyshev_heat_kernel(lap, s, m=order)

        for idx, node in enumerate(nodes):
            coeffs = np.exp(1j * np.outer(ts, heat[idx, :])).sum(axis=1)
            emb[node].extend(coeffs.real)
            emb[node].extend(coeffs.imag)

    return {n: np.asarray(v, dtype=float) for n, v in emb.items()}


def graphwave_entropy(
    embeddings: Dict[object, Iterable[float]]
) -> float:  # pragma: no cover - rarely used
    r"""Return GraphWave entropy based on embedding norms.

    The differential entropy used for autotuning is computed as

    .. math::

       H_{\text{wave}} = -\frac{1}{N} \sum_{u} \log \|\psi_u\|_2,

    where ``N`` is the number of embeddings and ``\psi_u`` the vector for node
    ``u``.  This avoids an expensive covariance estimate while providing a
    stable measure of spread.

    Parameters
    ----------
    embeddings:
        Mapping of nodes to embedding vectors.

    Returns
    -------
    float
        Estimated differential entropy.
    """

    arr = np.vstack([np.asarray(v, dtype=float) for v in embeddings.values()])
    if arr.size == 0:
        return 0.0
    norms = np.linalg.norm(arr, axis=1) + 1e-12
    return float(-np.log(norms).mean())


def embedding_entropy(
    embeddings: Dict[object, Iterable[float]]
) -> float:  # pragma: no cover - rarely used
    r"""Return differential entropy for an embedding dictionary.

    This is a generic variant of :func:`graphwave_entropy` that applies to any
    set of embedding vectors. The computation assumes a multivariate Gaussian
    distribution and uses the covariance determinant as

    .. math:: H = \tfrac{1}{2}\log\bigl((2\pi e)^d \det \Sigma\bigr).

    Parameters
    ----------
    embeddings:
        Mapping of arbitrary keys to embedding vectors.

    Returns
    -------
    float
        Estimated differential entropy of the embeddings.
    """

    arr = np.vstack([np.asarray(v, dtype=float) for v in embeddings.values()])
    if arr.size == 0:
        return 0.0
    cov = np.cov(arr, rowvar=False)
    d = cov.shape[0]
    sign, logdet = np.linalg.slogdet(cov + 1e-8 * np.eye(d))
    if sign <= 0:
        return float("nan")
    return 0.5 * (logdet + d * math.log(2 * math.pi * math.e))


def bottleneck_distance(
    g1: nx.Graph, g2: nx.Graph, dimension: int = 0, approx_epsilon: float | None = None
) -> float:  # pragma: no cover - requires gudhi
    """Return bottleneck distance between ``g1`` and ``g2`` diagrams.

    Parameters
    ----------
    g1, g2:
        Graphs to compare.
    dimension:
        Homology dimension used for the persistence diagrams.
    approx_epsilon:
        Additive error tolerated by :func:`gudhi.bottleneck_distance`. ``None``
        (the default) means use the smallest positive ``float`` for exact
        computation.

    Returns
    -------
    float
        Bottleneck distance between the diagrams of ``g1`` and ``g2`` in the
        chosen dimension.
    """

    if gd is None:
        raise RuntimeError("gudhi is required for persistence calculations")

    def _diagram(graph: nx.Graph) -> np.ndarray:
        st = gd.SimplexTree()
        mapping = {n: i for i, n in enumerate(graph.nodes())}
        for node, idx in mapping.items():
            st.insert([idx], filtration=0.0)
        for u, v in graph.edges():
            st.insert([mapping[u], mapping[v]], filtration=1.0)

        st.compute_persistence(persistence_dim_max=True)
        diag = np.array(st.persistence_intervals_in_dimension(dimension))
        if diag.size == 0:
            return np.empty((0, 2))
        diag = diag[np.isfinite(diag[:, 1])]
        return diag

    d1 = _diagram(g1)
    d2 = _diagram(g2)
    return float(gd.bottleneck_distance(d1, d2, e=approx_epsilon))


def persistence_wasserstein_distance(
    g1: nx.Graph, g2: nx.Graph, dimension: int = 0, order: int = 1
) -> float:  # pragma: no cover - requires gudhi and heavy computation
    """Return Wasserstein distance between ``g1`` and ``g2`` diagrams.

    Parameters
    ----------
    g1, g2:
        Graphs to compare.
    dimension:
        Homology dimension used for the persistence diagrams.
    order:
        Wasserstein order ``q`` controlling the exponent of the ground metric.

    Returns
    -------
    float
        Wasserstein distance between the diagrams of ``g1`` and ``g2`` in the
        chosen dimension.
    """

    if gd is None:
        raise RuntimeError("gudhi is required for persistence calculations")

    def _diagram(graph: nx.Graph) -> np.ndarray:
        st = gd.SimplexTree()
        mapping = {n: i for i, n in enumerate(graph.nodes())}
        for node, idx in mapping.items():
            st.insert([idx], filtration=0.0)
        for u, v in graph.edges():
            st.insert([mapping[u], mapping[v]], filtration=1.0)

        st.compute_persistence(persistence_dim_max=True)
        diag = np.array(st.persistence_intervals_in_dimension(dimension))
        if diag.size == 0:
            return np.empty((0, 2))
        diag = diag[np.isfinite(diag[:, 1])]
        return diag

    d1 = _diagram(g1)
    d2 = _diagram(g2)
    return float(gd.wasserstein_distance(d1, d2, order=order, internal_p=order))


def mdl_optimal_radius(counts: List[Tuple[int, int]]) -> int:
    """Return radius index minimizing a simple MDL criterion.

    Parameters
    ----------
    counts:
        Sequence of ``(radius, box_count)`` pairs returned by
        :func:`box_counting_dimension`. Must be ordered by increasing radius.

    Returns
    -------
    int
        Index of ``counts`` giving the minimal description length. ``counts``
        must contain at least two elements; otherwise 0 is returned.
    """

    if len(counts) < 2:
        return 0

    mdls = []
    for end in range(2, len(counts) + 1):
        sub = counts[:end]
        xs = [-math.log(float(r)) for r, _ in sub]
        ys = [math.log(float(n)) for _, n in sub]
        slope, intercept = np.polyfit(xs, ys, 1)
        pred = [slope * x + intercept for x in xs]
        rss = sum((y - p) ** 2 for y, p in zip(ys, pred))
        n = len(xs)
        k = 2  # slope and intercept
        mdl = n * math.log(rss / n + 1e-12) + k * math.log(n)
        mdls.append((mdl, end - 1))

    best = min(mdls, key=lambda x: x[0])[1]
    return best


def mdl_value(counts: List[Tuple[int, int]]) -> float:
    """Return MDL value for a sequence of ``(radius, box_count)`` pairs."""

    if len(counts) < 2:
        return 0.0

    xs = [-math.log(float(r)) for r, _ in counts]
    ys = [math.log(float(n)) for _, n in counts]
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = [slope * x + intercept for x in xs]
    rss = sum((y - p) ** 2 for y, p in zip(ys, pred))
    n = len(xs)
    k = 2  # slope and intercept
    return float(n * math.log(rss / n + 1e-12) + k * math.log(n))


def _slope(counts: List[Tuple[int, int]]) -> float:
    """Return slope of ``log(N_B)`` vs ``log(1/l_B)`` for ``counts``."""

    xs = [-math.log(float(r)) for r, _ in counts]
    ys = [math.log(float(n)) for _, n in counts]
    if len(xs) < 2:
        return 0.0
    slope, _ = np.polyfit(xs, ys, 1)
    return float(slope)


def dichotomic_radius(counts: List[Tuple[int, int]], target: float) -> int:
    """Return index whose slope is closest to ``target`` using dichotomy."""

    left = 1
    right = len(counts) - 1
    best = 1
    best_diff = abs(_slope(counts[: best + 1]) - target)
    while left <= right:
        mid = (left + right) // 2
        s = _slope(counts[: mid + 1])
        diff = abs(s - target)
        if diff < best_diff:
            best = mid
            best_diff = diff
        if s > target:
            left = mid + 1
        else:
            right = mid - 1
    return best


def spectral_dimension(
    graph: nx.Graph, times: Iterable[float]
) -> Tuple[float, List[Tuple[float, float]]]:  # pragma: no cover - heavy computation
    r"""Estimate the spectral dimension of ``graph`` using heat trace scaling.

    The heat trace :math:`\mathrm{Tr}(e^{-tL})` of the graph Laplacian
    asymptotically scales like ``t**(-d/2)`` where ``d`` is the spectral
    dimension.  By computing the heat trace for a range of ``times`` and
    fitting a line to ``log(trace)`` versus ``log(time)`` we obtain an estimate
    of ``d``.

    Parameters
    ----------
    graph:
        Input graph.
    times:
        Iterable of diffusion times ``t``. Larger values probe coarser
        structures of the graph.

    Returns
    -------
    float
        Estimated spectral dimension.
    list[tuple[float, float]]
        Pairs ``(t, trace)`` used for the fit.
    """

    nodes = list(graph.nodes())
    a = nx.to_numpy_array(graph, nodelist=nodes)
    lap = _laplacian(graph, normed=False)
    evals, _ = eigh(lap)

    traces: List[Tuple[float, float]] = []
    for t in times:
        trace = float(np.sum(np.exp(-t * evals)))
        traces.append((float(t), trace))

    xs = [math.log(t) for t, _ in traces]
    ys = [math.log(tr) for _, tr in traces]
    if len(xs) < 2:
        return 0.0, traces
    slope, _ = np.polyfit(xs, ys, 1)
    dim = -2 * slope
    return float(dim), traces


def embedding_box_counting_dimension(
    coords: Dict[object, Iterable[float]], radii: Iterable[float]
) -> Tuple[float, List[Tuple[float, int]]]:  # pragma: no cover - heavy computation
    """Estimate fractal dimension from point coordinates.

    Parameters
    ----------
    coords:
        Mapping of node IDs to embedding vectors.
    radii:
        Iterable of ball radii. Larger radii correspond to coarser covers.

    Returns
    -------
    float
        Estimated fractal dimension based on the slope of ``log(N_B)`` vs
        ``log(1/r)``.
    list[tuple[float, int]]
        ``(radius, count)`` pairs used for the estimation.
    """

    pts = np.asarray(list(coords.values()), dtype=float)
    counts: List[Tuple[float, int]] = []
    for r in radii:
        remaining = set(range(len(pts)))
        boxes = 0
        while remaining:
            i = remaining.pop()
            dists = np.linalg.norm(pts[list(remaining | {i})] - pts[i], axis=1)
            cover = {j for j, d in zip(list(remaining | {i}), dists) if d <= r}
            remaining.difference_update(cover)
            boxes += 1
        counts.append((float(r), boxes))

    xs = [-math.log(float(r)) for r, _ in counts]
    ys = [math.log(float(n)) for _, n in counts]
    if len(xs) < 2:
        return 0.0, counts
    slope, _ = np.polyfit(xs, ys, 1)
    return float(slope), counts


def latent_box_dimension(
    coords: Mapping[object, Iterable[float]],
    radii: Iterable[float],
    *,
    sample: int = 512,
) -> Tuple[float, List[Tuple[float, int]]]:
    """Return fast box-counting dimension of embedding coordinates.

    The coordinates are optionally subsampled to ``sample`` points and covered
    using a KD-tree for efficiency.
    """

    from scipy.spatial import cKDTree

    pts = np.asarray(list(coords.values()), dtype=float)
    if pts.size == 0:
        return 0.0, []
    if pts.shape[0] > sample:
        idx = np.random.choice(pts.shape[0], sample, replace=False)
        pts = pts[idx]
    tree = cKDTree(pts)
    counts: list[tuple[float, int]] = []
    for r in radii:
        seen = set()
        boxes = 0
        for i in range(len(pts)):
            if i in seen:
                continue
            neighbors = tree.query_ball_point(pts[i], r)
            seen.update(neighbors)
            boxes += 1
        counts.append((float(r), boxes))

    xs = [-math.log(float(r)) for r, _ in counts]
    ys = [math.log(float(n)) for _, n in counts]
    if len(xs) < 2:
        return 0.0, counts
    slope, _ = np.polyfit(xs, ys, 1)
    return float(slope), counts


def fractal_dim_embedding(
    coords: Mapping[object, Iterable[float]],
    radii: Iterable[float],
    *,
    sample: int = 512,
) -> float:
    """Estimate the fractal dimension of embedding coordinates.

    The function applies a box-counting procedure to the embedding vectors and
    returns the slope of ``\log N_B`` versus ``-\log r`` where ``N_B`` is the
    number of boxes of radius ``r`` covering the points:

    .. math::

        D_f = \frac{\log N_B(r)}{-\log r}

    Parameters
    ----------
    coords:
        Mapping of node identifiers to embedding vectors.
    radii:
        Iterable of cover radii used for the box-counting estimate.
    sample:
        Optional number of points to subsample before counting. Defaults to
        ``512``.

    Returns
    -------
    float
        Estimated fractal dimension of the embedding space.
    """

    dim, _ = latent_box_dimension(coords, radii, sample=sample)
    return dim


def laplacian_spectrum(
    graph: nx.Graph, *, normed: bool = True
) -> np.ndarray:  # pragma: no cover - heavy computation
    """Return the Laplacian eigenvalues of ``graph``.

    Parameters
    ----------
    graph:
        Input graph.
    normed:
        Whether to compute the normalized Laplacian. Defaults to ``True``.

    Returns
    -------
    numpy.ndarray
        Array of eigenvalues sorted in ascending order.
    """

    a = nx.to_numpy_array(graph, nodelist=list(graph.nodes()))
    lap = _laplacian(graph, normed=normed)
    evals = eigh(lap, eigvals_only=True)
    return np.sort(evals)


def spectral_entropy(
    graph: nx.Graph, *, normed: bool = True
) -> float:  # pragma: no cover - heavy computation
    """Return the Shannon entropy of the Laplacian spectrum.

    The eigenvalues of the Laplacian are normalized to form a probability
    distribution before computing the entropy.

    Parameters
    ----------
    graph:
        Input graph.
    normed:
        Whether to use the normalized Laplacian. Defaults to ``True``.

    Returns
    -------
    float
        Shannon entropy of the normalized spectrum.
    """

    evals = laplacian_spectrum(graph, normed=normed)
    total = float(np.sum(evals))
    if total == 0:
        return 0.0
    probs = evals / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def spectral_gap(
    graph: nx.Graph, *, normed: bool = True
) -> float:  # pragma: no cover - heavy computation
    """Return the spectral gap of ``graph``.

    The spectral gap is the difference between the two smallest eigenvalues
    of the (normalized) Laplacian. For a connected graph with a normalized
    Laplacian this is simply ``lambda_2``.

    Parameters
    ----------
    graph:
        Input graph.
    normed:
        Whether to use the normalized Laplacian. Defaults to ``True``.

    Returns
    -------
    float
        The spectral gap value. Returns ``0.0`` if there are fewer than two
        eigenvalues.
    """

    evals = laplacian_spectrum(graph, normed=normed)
    if len(evals) < 2:
        return 0.0
    return float(evals[1] - evals[0])


def laplacian_energy(
    graph: nx.Graph, *, normed: bool = True
) -> float:  # pragma: no cover - heavy computation
    r"""Return the Laplacian energy of ``graph``.

    The Laplacian energy is defined as

    .. math:: \sum_{i} |\lambda_i - 2m/n|

    where :math:`\lambda_i` are the Laplacian eigenvalues, ``m`` is the number
    of edges and ``n`` is the number of nodes.

    Parameters
    ----------
    graph:
        Input graph.
    normed:
        Whether to use the normalized Laplacian. Defaults to ``True``.

    Returns
    -------
    float
        Laplacian energy value.
    """

    evals = laplacian_spectrum(graph, normed=normed)
    m = graph.number_of_edges()
    n = graph.number_of_nodes()
    if n == 0:
        return 0.0
    avg = 2 * m / n
    return float(np.sum(np.abs(evals - avg)))


def spectral_density(
    graph: nx.Graph, bins: int = 50, *, normed: bool = True
) -> Tuple[np.ndarray, np.ndarray]:  # pragma: no cover - heavy computation
    """Return the density of Laplacian eigenvalues.

    Parameters
    ----------
    graph:
        Input graph.
    bins:
        Number of histogram bins.
    normed:
        Whether to use the normalized Laplacian.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Histogram counts and the corresponding bin edges.
    """

    evals = laplacian_spectrum(graph, normed=normed)
    hist, edges = np.histogram(evals, bins=bins, density=True)
    return hist.astype(float), edges.astype(float)


def graph_lacunarity(graph: nx.Graph, radius: int = 1) -> float:
    r"""Return lacunarity of ``graph`` for neighborhood radius ``radius``.

    The lacunarity measures the heterogeneity of mass distribution
    at a given scale. For each node we count the number of nodes
    within ``radius`` hops and compute

    .. math:: \Lambda = \frac{\mathrm{Var}(M)}{\mathrm{E}[M]^2} + 1,

    where :math:`M` denotes the local mass.
    """

    masses = [
        nx.ego_graph(graph, n, radius=radius).number_of_nodes() for n in graph.nodes()
    ]
    if not masses:
        return 0.0
    arr = np.asarray(masses, dtype=float)
    mean = arr.mean()
    if mean == 0:
        return 0.0
    return float(arr.var() / (mean * mean) + 1.0)


def graph_fourier_transform(
    graph: nx.Graph, signal: Dict[int, float] | np.ndarray, *, normed: bool = True
) -> np.ndarray:
    """Return the graph Fourier transform of ``signal``.

    Parameters
    ----------
    graph:
        Input graph.
    signal:
        Mapping from node to value or array of values ordered by ``graph.nodes``.
    normed:
        Whether to use the normalized Laplacian.

    Returns
    -------
    numpy.ndarray
        Fourier coefficients ordered by increasing Laplacian eigenvalue.
    """

    nodes = list(graph.nodes())
    lap = _laplacian(graph, normed=normed)
    evals, evecs = eigh(lap)
    if isinstance(signal, dict):
        vec = np.array([signal[n] for n in nodes], dtype=float)
    else:
        vec = np.asarray(signal, dtype=float)
    return evecs.T @ vec


def inverse_graph_fourier_transform(
    graph: nx.Graph, coeffs: np.ndarray, *, normed: bool = True
) -> np.ndarray:
    """Return the inverse graph Fourier transform of ``coeffs``."""

    nodes = list(graph.nodes())
    lap = _laplacian(graph, normed=normed)
    _, evecs = eigh(lap)
    return evecs @ np.asarray(coeffs, dtype=float)


def fractal_information_metrics(
    graph: nx.Graph, radii: Iterable[int], *, max_dim: int = 1
) -> Dict[str, Any]:
    """Return fractal dimension and persistence entropies.

    Parameters
    ----------
    graph:
        Input graph.
    radii:
        Iterable of box radii used for the covering.
    max_dim:
        Highest homology dimension for which to compute the entropy.

    Returns
    -------
    dict
        Mapping with keys ``dimension`` and ``entropy`` (per homology dimension).
    """

    dim, _ = box_counting_dimension(graph, radii)
    entropies: Dict[int, float] = {}
    if gd is not None and gr is not None:
        for d in range(max_dim + 1):
            try:
                entropies[d] = persistence_entropy(graph, dimension=d)
            except Exception:  # pragma: no cover - optional dep failure
                entropies[d] = float("nan")
    else:  # pragma: no cover - optional dep missing
        entropies = {d: float("nan") for d in range(max_dim + 1)}

    return {"dimension": dim, "entropy": entropies}


def fractal_information_density(
    graph: nx.Graph, radii: Iterable[int], *, max_dim: int = 1
) -> float:
    """Return a simple information density from fractal dimension and entropy.

    The density is defined as ``dimension / (1 + sum(entropies))`` so that
    higher entropies lower the returned value. It is meant as a lightweight
    indicator of how much structural information is carried per fractal degree.
    """

    metrics = fractal_information_metrics(graph, radii, max_dim=max_dim)
    dim = metrics["dimension"]
    ent_sum = float(sum(metrics["entropy"].values()))
    return dim / (1.0 + ent_sum)


def fractal_level_coverage(graph: nx.Graph) -> float:  # pragma: no cover - rarely used
    """Return fraction of nodes annotated with a ``fractal_level``."""

    total = graph.number_of_nodes()
    if total == 0:
        return 0.0
    covered = sum(1 for _, data in graph.nodes(data=True) if "fractal_level" in data)
    return covered / float(total)


def diversification_score(
    global_graph: nx.Graph,
    batch_graph: nx.Graph,
    radii: Iterable[int],
    *,
    max_dim: int = 1,
    dimension: int = 0,
) -> float:  # pragma: no cover - rarely used
    """Return a diversification score mixing MDL and bottleneck distance.

    The score is the sum of the MDL difference between ``global_graph`` and
    ``batch_graph`` and the bottleneck distance of their persistence diagrams in
    ``dimension``. Lower scores indicate higher redundancy of ``batch_graph``
    with respect to ``global_graph``.
    """

    _, global_counts = box_counting_dimension(global_graph, radii)
    _, batch_counts = box_counting_dimension(batch_graph, radii)
    mdl_global = mdl_value(global_counts)
    mdl_batch = mdl_value(batch_counts)

    try:
        dist = bottleneck_distance(global_graph, batch_graph, dimension=dimension)
    except Exception:
        dist = float("nan")

    return float((mdl_global - mdl_batch) + dist)


def poincare_embedding(
    graph: nx.Graph,
    dim: int = 2,
    negative: int = 5,
    epochs: int = 50,
    learning_rate: float = 0.1,
    burn_in: int = 10,
) -> Dict[int, np.ndarray]:  # pragma: no cover - heavy gensim dependency
    """Return Poincar\u00e9 embeddings for ``graph``.

    Parameters
    ----------
    graph:
        Input graph whose edges define the hierarchy.
    dim:
        Dimension of the hyperbolic embedding space.
    negative:
        Number of negative samples used during training.
    epochs:
        Number of training epochs.
    learning_rate:
        Learning rate for the optimizer.
    burn_in:
        Number of burn-in epochs before using negative sampling.

    Returns
    -------
    dict
        Mapping of node to embedding vector of length ``dim``.
    """

    try:  # pragma: no cover - dependency not always present
        from gensim.models.poincare import PoincareModel
    except Exception as e:  # pragma: no cover - dependency missing
        raise RuntimeError("gensim is required for Poincare embeddings") from e

    edges = [(str(u), str(v)) for u, v in graph.edges()]
    model = PoincareModel(
        edges,
        size=dim,
        negative=negative,
        burn_in=burn_in,
        alpha=learning_rate,
    )
    model.train(epochs)

    embeddings: Dict[int, np.ndarray] = {}
    for node in graph.nodes():
        key = str(node)
        if key in model.kv:
            embeddings[node] = np.asarray(model.kv[key], dtype=float)

    if embeddings:
        r_mean = float(np.mean([np.linalg.norm(v) for v in embeddings.values()]))
        if r_mean > 0.9:
            logging.getLogger(__name__).warning(
                "Poincare embedding crowding detected: r_mean=%.3f", r_mean
            )
    return embeddings


def fractalize_graph(
    graph: nx.Graph, radius: int
) -> Tuple[nx.Graph, Dict[object, int]]:  # pragma: no cover - rarely used
    """Return a coarse-grained graph obtained via box covering.

    Parameters
    ----------
    graph:
        Input graph to coarse-grain.
    radius:
        Radius ``l_B`` used for the covering.

    Returns
    -------
    tuple
        A tuple ``(G, mapping)`` where ``G`` is the coarse-grained graph and
        ``mapping`` maps each original node to its box index.
    """

    boxes = box_cover(graph, radius)
    mapping: Dict[object, int] = {}
    coarse = nx.Graph()
    for idx, box in enumerate(boxes):
        coarse.add_node(idx)
        for n in box:
            mapping[n] = idx

    for u, v in graph.edges():
        b1 = mapping[u]
        b2 = mapping[v]
        if b1 != b2:
            coarse.add_edge(b1, b2)

    return coarse, mapping


def fractalize_optimal(
    graph: nx.Graph, radii: Iterable[int]
) -> Tuple[nx.Graph, Dict[object, int], int]:  # pragma: no cover - rarely used
    """Coarse-grain ``graph`` using the MDL-optimal radius.

    Parameters
    ----------
    graph:
        Input graph to coarse-grain.
    radii:
        Candidate box radii ``l_B`` used to search for the optimal cover.

    Returns
    -------
    tuple
        ``(G, mapping, radius)`` where ``G`` is the coarse-grained graph,
        ``mapping`` maps each original node to its box index, and ``radius``
        is the chosen radius.
    """

    _, counts = box_counting_dimension(graph, radii)
    if not counts:
        raise ValueError("radii must not be empty")
    idx = mdl_optimal_radius(counts)
    radius = counts[idx][0]
    coarse, mapping = fractalize_graph(graph, radius)
    return coarse, mapping, radius


def build_fractal_hierarchy(
    graph: nx.Graph, radii: Iterable[int], *, max_levels: int = 5
) -> List[Tuple[nx.Graph, Dict[object, int], int]]:  # pragma: no cover - rarely used
    """Return a hierarchy of coarse graphs using MDL-optimal radii.

    Parameters
    ----------
    graph:
        Input graph to coarse-grain recursively.
    radii:
        Candidate box radii ``l_B`` used to search for the optimal cover at each
        level.
    max_levels:
        Maximum number of coarse-graining iterations. The process stops early if
        the graph can no longer be reduced.

    Returns
    -------
    list
        Sequence ``[(G1, mapping1, r1), (G2, mapping2, r2), ...]`` describing
        the hierarchy from fine to coarse. ``Gi`` is the graph at level ``i`` and
        ``mappingi`` maps nodes of ``G_{i-1}`` to boxes of ``Gi``.
    """

    levels: List[Tuple[nx.Graph, Dict[object, int], int]] = []
    current = graph
    for _ in range(max_levels):
        coarse, mapping, radius = fractalize_optimal(current, radii)
        levels.append((coarse, mapping, radius))
        if (
            coarse.number_of_nodes() >= current.number_of_nodes()
            or coarse.number_of_nodes() <= 1
        ):
            break
        current = coarse
    return levels


def build_mdl_hierarchy(
    graph: nx.Graph,
    radii: Iterable[int],
    *,
    max_levels: int = 5,
    slope_tol: float = 0.1,
) -> List[Tuple[nx.Graph, Dict[object, int], int]]:  # pragma: no cover - rarely used
    """Return a hierarchy using MDL to stop when description length grows."""

    levels: List[Tuple[nx.Graph, Dict[object, int], int]] = []
    current = graph
    prev_mdl = float("inf")
    target_dim = None
    for _ in range(max_levels):
        dim, counts = box_counting_dimension(current, radii)
        if not counts:
            break
        idx = mdl_optimal_radius(counts)
        if target_dim is not None:
            cand = dichotomic_radius(counts, target_dim)
            if abs(_slope(counts[: cand + 1]) - target_dim) <= slope_tol:
                idx = cand
        mdl_curr = mdl_value(counts[: idx + 1])
        if mdl_curr >= prev_mdl:
            break
        prev_mdl = mdl_curr
        radius = counts[idx][0]
        coarse, mapping = fractalize_graph(current, radius)
        levels.append((coarse, mapping, radius))
        if (
            coarse.number_of_nodes() >= current.number_of_nodes()
            or coarse.number_of_nodes() <= 1
        ):
            break
        target_dim = _slope(counts[: idx + 1])
        current = coarse
    return levels


def minimize_bottleneck_distance(
    graph: nx.Graph,
    target: nx.Graph,
    *,
    dimension: int = 1,
    epsilon: float = 0.0,
    max_iter: int = 100,
    seed: int | None = None,
) -> Tuple[nx.Graph, float]:  # pragma: no cover - stochastic optimization
    """Return a graph whose topology approximates ``target``.

    The function greedily edits edges to minimize the bottleneck distance
    between ``graph`` and ``target`` in ``dimension``. Edges are added or
    removed at random and a change is kept if it improves the distance by at
    least ``epsilon``.

    Parameters
    ----------
    graph:
        Input graph to modify. A copy is used internally.
    target:
        Reference graph defining the desired topology.
    dimension:
        Homology dimension for the bottleneck distance.
    epsilon:
        Minimum improvement required to accept a modification.
    max_iter:
        Maximum number of edge edits to attempt.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    tuple
        ``(G, dist)`` where ``G`` is the optimized graph and ``dist`` is the
        final bottleneck distance to ``target``.
    """

    rng = random.Random(seed)  # nosec - not used for cryptography

    best = graph.copy()
    best_dist = bottleneck_distance(best, target, dimension=dimension)

    for _ in range(max_iter):
        candidate = best.copy()
        if rng.random() < 0.5 and list(nx.non_edges(candidate)):
            u, v = rng.choice(list(nx.non_edges(candidate)))
            candidate.add_edge(u, v)
        elif candidate.number_of_edges() > 0:
            u, v = rng.choice(list(candidate.edges()))
            candidate.remove_edge(u, v)
        else:
            continue

        dist = bottleneck_distance(candidate, target, dimension=dimension)
        if dist + epsilon < best_dist:
            best = candidate
            best_dist = dist
        if best_dist <= epsilon:
            break

    return best, float(best_dist)


def hyperbolic_distance(
    x: np.ndarray, y: np.ndarray
) -> float:  # pragma: no cover - geometric routine
    """Return the Poincar\u00e9 distance between ``x`` and ``y``."""

    x2 = np.dot(x, x)
    y2 = np.dot(y, y)
    diff = np.linalg.norm(x - y) ** 2
    denom = (1 - x2) * (1 - y2)
    if denom <= 0:
        return float("inf")
    cosh = 1 + 2 * diff / denom
    return float(np.arccosh(max(1.0, cosh)))


def hyperbolic_nearest_neighbors(
    embeddings: Dict[object, Iterable[float]], k: int = 5
) -> Dict[object, List[tuple[object, float]]]:  # pragma: no cover - geometric routine
    """Return ``k`` nearest neighbors for each node in hyperbolic space."""

    vecs = {n: np.asarray(v, dtype=float) for n, v in embeddings.items()}
    neighbors: Dict[object, List[tuple[object, float]]] = {}
    for u, vec_u in vecs.items():
        dists = []
        for v, vec_v in vecs.items():
            if u == v:
                continue
            dist = hyperbolic_distance(vec_u, vec_v)
            dists.append((v, dist))
        dists.sort(key=lambda t: t[1])
        neighbors[u] = dists[:k]
    return neighbors


def hyperbolic_reasoning(
    embeddings: Dict[object, Iterable[float]],
    start: object,
    goal: object,
    *,
    max_steps: int = 5,
) -> List[object]:  # pragma: no cover - geometric routine
    """Return a greedy path from ``start`` to ``goal`` in hyperbolic space."""

    vecs = {n: np.asarray(v, dtype=float) for n, v in embeddings.items()}
    if start not in vecs or goal not in vecs:
        raise ValueError("start or goal missing from embeddings")

    current = start
    path = [current]
    for _ in range(max_steps):
        if current == goal:
            break

        candidates = [n for n in vecs if n != current]
        if not candidates:
            break
        next_node = min(
            candidates,
            key=lambda n: hyperbolic_distance(vecs[current], vecs[n])
            + hyperbolic_distance(vecs[n], vecs[goal]),
        )
        if next_node == current or next_node in path:
            break
        path.append(next_node)
        current = next_node
    return path


def hyperbolic_hypergraph_reasoning(
    embeddings: Dict[object, Iterable[float]],
    hyperedges: Iterable[object],
    start: object,
    goal: object,
    *,
    penalty: float = 1.0,
    max_steps: int = 5,
) -> List[object]:  # pragma: no cover - geometric routine
    """Return a greedy path in hyperbolic space considering hyperedges."""

    vecs = {n: np.asarray(v, dtype=float) for n, v in embeddings.items()}
    if start not in vecs or goal not in vecs:
        raise ValueError("start or goal missing from embeddings")
    hyper_set = set(hyperedges)

    current = start
    path = [current]
    for _ in range(max_steps):
        if current == goal:
            break

        candidates = [n for n in vecs if n != current]
        if not candidates:
            break
        next_node = min(
            candidates,
            key=lambda n: hyperbolic_distance(vecs[current], vecs[n])
            + hyperbolic_distance(vecs[n], vecs[goal])
            + (penalty if n in hyper_set else 0.0),
        )
        if next_node == current or next_node in path:
            break
        path.append(next_node)
        current = next_node
    return path


def hyperbolic_multi_curvature_reasoning(
    embeddings: Dict[float, Dict[object, Iterable[float]]],
    start: object,
    goal: object,
    *,
    weights: Optional[Dict[float, float]] = None,
    max_steps: int = 5,
) -> List[object]:  # pragma: no cover - geometric routine
    """Return a greedy path combining several hyperbolic curvatures."""

    if not embeddings:
        return []
    base_key = next(iter(embeddings))
    nodes = set(embeddings[base_key])
    if start not in nodes or goal not in nodes:
        raise ValueError("start or goal missing from embeddings")

    weights = weights or {c: 1.0 for c in embeddings}
    vecs = {
        c: {n: np.asarray(v, dtype=float) for n, v in embs.items()}
        for c, embs in embeddings.items()
    }

    def _dist(u: object, v: object) -> float:
        return sum(
            weights.get(c, 1.0) * hyperbolic_distance(vecs[c][u], vecs[c][v])
            for c in vecs
            if u in vecs[c] and v in vecs[c]
        )

    current = start
    path = [current]
    for _ in range(max_steps):
        if current == goal:
            break
        candidates = [n for n in nodes if n != current]
        if not candidates:
            break
        next_node = min(candidates, key=lambda n: _dist(current, n) + _dist(n, goal))
        if next_node == current or next_node in path:
            break
        path.append(next_node)
        current = next_node

    return path


def fractal_net_prune(
    embeddings: Dict[object, Iterable[float]],
    *,
    tol: float = 1e-3,
) -> Tuple[
    Dict[int, np.ndarray], Dict[object, int]
]:  # pragma: no cover - iterative routine
    """Return pruned embedding centers and node mapping.

    The function greedily merges embedding vectors whose Euclidean distance is
    below ``tol``. Each node is assigned to a cluster represented by the mean
    of its members. The returned dictionary maps cluster index to centroid
    vector, while ``mapping`` tells which cluster each node belongs to.
    """

    centers: List[np.ndarray] = []
    mapping: Dict[object, int] = {}

    for node, vec in embeddings.items():
        arr = np.asarray(vec, dtype=float)
        assigned = False
        for idx, c in enumerate(centers):
            if np.linalg.norm(arr - c) <= tol:
                centers[idx] = (c + arr) / 2.0
                mapping[node] = idx
                assigned = True
                break
        if not assigned:
            centers.append(arr)
            mapping[node] = len(centers) - 1

    pruned = {i: centers[i] for i in range(len(centers))}
    return pruned, mapping


def fractalnet_compress(
    embeddings: Dict[object, Iterable[float]],
    levels: Dict[object, int],
) -> Dict[int, np.ndarray]:  # pragma: no cover - iterative routine
    """Return averaged embeddings for each fractal level.

    Parameters
    ----------
    embeddings:
        Mapping of node id to embedding vector.
    levels:
        Mapping of node id to ``fractal_level`` integer.

    Returns
    -------
    dict
        Mapping ``level -> centroid`` computed as the mean of embeddings
        belonging to that level. Levels with no embeddings are omitted.
    """
    groups: Dict[int, List[np.ndarray]] = {}
    for node, vec in embeddings.items():
        lvl = levels.get(node)
        if lvl is None:
            continue
        groups.setdefault(int(lvl), []).append(np.asarray(vec, dtype=float))
    return {lvl: np.mean(vs, axis=0) for lvl, vs in groups.items() if vs}


def inject_graphrnn_subgraph(
    graph: nx.Graph, num_nodes: int, num_edges: int
) -> list[object]:  # pragma: no cover - external dependency
    """Inject a GraphRNN motif into ``graph`` and return created nodes.

    The routine tries to use :class:`GraphRNN_Lite` from ``torch_geometric_temporal``
    to sample a small graph. When the dependency is missing a simple NetworkX
    approximation is used instead.
    """

    try:  # pragma: no cover - optional heavy dependency
        from torch_geometric_temporal.nn.models import GraphRNN_Lite  # type: ignore

        model = GraphRNN_Lite(input_size=num_nodes, hidden_size=num_nodes)
        motif = model.sample(num_nodes)
    except Exception:
        from .generation import generate_graph_rnn_like as _gen

        motif = _gen(num_nodes, num_edges)

    base = max(graph.nodes, default=-1) + 1
    mapping = {n: base + i for i, n in enumerate(motif.nodes())}
    graph.add_nodes_from((m, {"label": "RNN_PATCH"}) for m in mapping.values())
    for u, v, data in motif.edges(data=True):
        graph.add_edge(mapping[u], mapping[v], **data)
    return list(mapping.values())


def inject_and_validate(
    graph: nx.Graph,
    num_nodes: int,
    num_edges: int,
    *,
    rollback: bool = True,
    driver: "Driver | None" = None,
) -> float:  # pragma: no cover - external dependency
    """Inject GraphRNN motif and return sheaf consistency score.

    Parameters
    ----------
    graph:
        Graph to augment.
    num_nodes, num_edges:
        Size of the motif to generate and insert.
    rollback:
        If ``True`` and the sheaf score is below ``0.8`` the injected nodes are
        removed again.
    """

    nodes = inject_graphrnn_subgraph(graph, num_nodes, num_edges)
    from .sheaf import validate_section

    if driver is not None and Driver is not None:
        with driver.session() as session:
            for n in nodes:
                session.run("MERGE (p:RNN_PATCH {id:$id})", id=n)
            for u, v in graph.subgraph(nodes).edges():
                session.run(
                    "MATCH (a:RNN_PATCH {id:$u}), (b:RNN_PATCH {id:$v}) MERGE (a)-[:RNN_EDGE]->(b)",
                    u=u,
                    v=v,
                )

    score = validate_section(graph, nodes)
    if rollback and score < 0.8:
        graph.remove_nodes_from(nodes)
        if driver is not None and Driver is not None:
            with driver.session() as session:
                session.run(
                    "MATCH (n:RNN_PATCH) WHERE n.id IN $ids DETACH DELETE n",
                    ids=nodes,
                )
    return score


def tpl_motif_injection(
    graph: nx.Graph, cfg: Mapping[str, Any], driver: "Driver | None" = None
) -> float:  # pragma: no cover - external dependency
    """Generate and inject a GraphRNN motif based on configuration."""

    size = int(cfg.get("tpl", {}).get("rnn_size", 64))
    return inject_and_validate(
        graph, size, max(1, size - 1), rollback=False, driver=driver
    )


def bootstrap_sigma_db(
    graph: nx.Graph, radii: Iterable[int]
) -> float:  # pragma: no cover - stochastic routine
    r"""Return bootstrap standard deviation of the fractal dimension.

    This mirrors the COLOUR-box GPU estimation but uses
    simple NetworkX operations. Thirty random 80% subgraphs of ``graph``
    are sampled. The fractal dimension of each is computed with
    :func:`colour_box_dimension`. The standard deviation

    .. math::

       \sigma_{d_B}=\sqrt{\tfrac1{29}\sum_i(d_B^i-\bar d_B)^2}

    is stored in ``graph.graph['fractal_sigma']``. A value above ``0.02``
    will later increase the autotuning cost (see :mod:`datacreek.analysis.autotune`).
    """

    dims: list[float] = []
    nodes = list(graph.nodes())
    for _ in range(30):
        sample = random.sample(nodes, max(1, int(0.8 * len(nodes))))
        sub = graph.subgraph(sample)
        dim, _ = colour_box_dimension(sub, radii)
        dims.append(dim)

    if not dims:
        sigma = 0.0
    else:
        mean = float(np.mean(dims))
        sigma = float(
            np.sqrt(sum((d - mean) ** 2 for d in dims) / max(1, len(dims) - 1))
        )

    graph.graph["fractal_sigma"] = sigma
    logging.getLogger(__name__).info("fractal_sigma=%.4f", sigma)
    return sigma
