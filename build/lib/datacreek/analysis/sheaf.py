from __future__ import annotations

import networkx as nx
import numpy as np


def sheaf_laplacian(graph: nx.Graph, *, edge_attr: str = "sheaf_sign") -> np.ndarray:
    """Return the sheaf Laplacian matrix of a signed graph.

    Each edge may carry a sign stored in ``edge_attr``. Missing attributes
    default to ``+1``. The graph is treated as undirected.
    """
    nodes = list(graph.nodes())
    index = {n: i for i, n in enumerate(nodes)}
    L = np.zeros((len(nodes), len(nodes)))
    for u, v, data in graph.to_undirected().edges(data=True):
        sign = data.get(edge_attr, 1.0)
        i, j = index[u], index[v]
        L[i, i] += 1
        L[j, j] += 1
        L[i, j] -= sign
        L[j, i] -= sign
    return L


def sheaf_incidence_matrix(
    graph: nx.Graph, *, edge_attr: str = "sheaf_sign"
) -> np.ndarray:
    r"""Return incidence matrix ``\delta`` for ``graph``.

    Parameters
    ----------
    graph:
        Input graph whose edges may carry ``edge_attr`` signs.
    edge_attr:
        Name of the edge attribute storing restriction signs.

    Returns
    -------
    numpy.ndarray
        Incidence matrix with one column per (undirected) edge.
    """

    nodes = list(graph.nodes())
    edges = list(graph.to_undirected().edges(data=True))
    idx = {n: i for i, n in enumerate(nodes)}
    B = np.zeros((len(nodes), len(edges)), dtype=int)
    for j, (u, v, data) in enumerate(edges):
        sign = int(data.get(edge_attr, 1))
        B[idx[u], j] = 1
        B[idx[v], j] = -sign
    return B


def sheaf_convolution(
    graph: nx.Graph,
    features: dict[object, np.ndarray],
    *,
    edge_attr: str = "sheaf_sign",
    alpha: float = 0.1,
) -> dict[object, np.ndarray]:
    """Return one step of sheaf convolution on ``features``.

    The update rule is ``X - alpha * L @ X`` where ``L`` is the sheaf
    Laplacian obtained from ``edge_attr``. ``features`` must map each
    node to a numeric vector of equal length.
    """

    nodes = list(graph.nodes())
    if not nodes:
        return {}

    L = sheaf_laplacian(graph, edge_attr=edge_attr)
    dim = len(next(iter(features.values())))
    X = np.zeros((len(nodes), dim))
    for i, n in enumerate(nodes):
        if n in features:
            X[i] = np.asarray(features[n], dtype=float)
    Y = X - alpha * L @ X
    return {n: Y[i] for i, n in enumerate(nodes)}


def sheaf_neural_network(
    graph: nx.Graph,
    features: dict[object, np.ndarray],
    *,
    layers: int = 2,
    alpha: float = 0.1,
    edge_attr: str = "sheaf_sign",
) -> dict[object, np.ndarray]:
    """Return node features after a simple sheaf neural network.

    Parameters
    ----------
    graph:
        Input graph with optional sign information on edges.
    features:
        Initial node features as a mapping ``node -> vector``.
    layers:
        Number of sheaf convolution layers to apply.
    alpha:
        Step size used by :func:`sheaf_convolution`.
    edge_attr:
        Name of the edge attribute storing the sheaf sign.

    Returns
    -------
    dict
        Mapping of nodes to their updated feature vectors after ``layers``
        convolutions with a ReLU nonlinearity.
    """

    out = {n: np.asarray(v, dtype=float) for n, v in features.items()}
    for _ in range(max(1, layers)):
        out = sheaf_convolution(graph, out, edge_attr=edge_attr, alpha=alpha)
        for n in out:
            out[n] = np.maximum(out[n], 0.0)
    return out


def sheaf_first_cohomology(
    graph: nx.Graph, *, edge_attr: str = "sheaf_sign", tol: float = 1e-5
) -> int:
    """Return the dimension of the first sheaf cohomology group ``H^1``.

    The kernel of the sheaf Laplacian approximates the space of harmonic
    1-cochains. We count eigenvalues below ``tol`` as zero modes.
    """

    L = sheaf_laplacian(graph, edge_attr=edge_attr)
    vals = np.linalg.eigvalsh(L)
    return int(np.sum(vals < tol))


def resolve_sheaf_obstruction(
    graph: nx.Graph,
    *,
    edge_attr: str = "sheaf_sign",
    max_iter: int = 10,
) -> int:
    """Try to reduce :math:`H^1` by flipping edge signs.

    The Huang-Chen (2024) corrector is approximated by greedily toggling
    edge signs whenever it decreases the first cohomology dimension.

    Parameters
    ----------
    graph:
        Input graph whose edges may carry ``edge_attr`` signs.
    edge_attr:
        Name of the edge attribute storing the sheaf sign (``+1`` or ``-1``).
    max_iter:
        Maximum number of sign flips to attempt.

    Returns
    -------
    int
        The resulting dimension of :math:`H^1` after corrections.
    """

    h1 = sheaf_first_cohomology(graph, edge_attr=edge_attr, tol=1e-5)
    for _ in range(max_iter):
        if h1 == 0:
            break
        improved = False
        for u, v, data in graph.edges(data=True):
            current = data.get(edge_attr, 1.0)
            candidate = -float(current)
            if candidate == current:
                continue
            data[edge_attr] = candidate
            new_h1 = sheaf_first_cohomology(graph, edge_attr=edge_attr, tol=1e-5)
            if new_h1 <= h1:
                h1 = new_h1
                improved = True
                break
            data[edge_attr] = current
        if not improved:
            break
    return h1


def sheaf_consistency_score(graph: nx.Graph, *, edge_attr: str = "sheaf_sign") -> float:
    """Return a [0, 1] score measuring sheaf consistency.

    The score is computed as ``1 / (1 + H^1)``, where ``H^1`` is the
    dimension of the first sheaf cohomology group. A value close to ``1``
    indicates high consistency of the sheaf structure.
    """

    h1 = sheaf_first_cohomology(graph, edge_attr=edge_attr, tol=1e-5)
    return 1.0 / (1.0 + float(h1))


def sheaf_first_cohomology_blocksmith(
    graph: nx.Graph,
    *,
    edge_attr: str = "sheaf_sign",
    block_size: int = 40000,
    tol: float = 1e-5,
    lam_thresh: float | None = None,
) -> int:
    """Approximate :math:`H^1` using a block-Smith reduction.

    The Laplacian is processed in chunks of ``block_size`` so that very
    large graphs do not require dense factorizations. When the number of
    nodes is below ``block_size`` the routine falls back to an exact
    eigen-decomposition. Otherwise it uses ``scipy.sparse.linalg.eigsh``
    to estimate the smallest eigenvalues.
    """

    delta = sheaf_incidence_matrix(graph, edge_attr=edge_attr)
    n = delta.shape[0]
    if n == 0:
        return 0

    if lam_thresh is None:
        from ..utils.config import load_config

        cfg = load_config()
        lam_thresh = float(cfg.get("sheaf", {}).get("lam_thresh", 1.0))

    if n == 0:
        return 0

    if n <= block_size:
        Delta = delta.T @ delta
        vals = np.linalg.eigvalsh(Delta)
        if any(float(v) > lam_thresh for v in np.sort(np.asarray(vals))[:5]):
            return 0
        return int(np.sum(vals < tol))

    try:  # pragma: no cover - optional dependency
        return block_smith(
            delta.astype(int), block_size=block_size, lam_thresh=lam_thresh
        )
    except Exception:  # fall back to dense eigendecomposition
        Delta = delta.T @ delta
        vals = np.linalg.eigvalsh(Delta)
        return int(np.sum(vals < tol))


def sheaf_consistency_score_batched(
    graph: nx.Graph,
    batches: Iterable[Iterable[object]],
    *,
    edge_attr: str = "sheaf_sign",
) -> list[float]:
    """Return sheaf consistency scores for several node batches.

    Each batch defines an induced subgraph on which
    :func:`sheaf_consistency_score` is evaluated. This allows processing
    large graphs in manageable chunks.
    """

    scores: list[float] = []
    for nodes in batches:
        sub = graph.subgraph(nodes)
        scores.append(sheaf_consistency_score(sub, edge_attr=edge_attr))
    return scores


def spectral_bound_exceeded(
    graph: nx.Graph, k: int, tau: float, *, edge_attr: str = "sheaf_sign"
) -> bool:
    r"""Return True if the k-th sheaf Laplacian eigenvalue exceeds ``tau``.

    Parameters
    ----------
    graph : nx.Graph
        Input graph with sheaf structure.
    k : int
        Index of the eigenvalue (1-indexed).
    tau : float
        Threshold for early stopping.
    edge_attr : str, optional
        Edge attribute storing restriction signs.

    Returns
    -------
    bool
        ``True`` if :math:`\lambda_k^\mathcal{F} > \tau`, ``False`` otherwise.
    """
    L = sheaf_laplacian(graph, edge_attr=edge_attr)
    if L.size == 0:
        return False
    n = L.shape[0]
    try:
        import scipy.sparse as sp  # pragma: no cover - optional
        from scipy.sparse.linalg import eigsh

        vals = eigsh(
            sp.csr_matrix(L), k=min(k, n - 1), which="LM", return_eigenvectors=False
        )
    except Exception:  # fallback to dense eigendecomposition
        vals = np.linalg.eigvalsh(L)
    vals = np.sort(np.asarray(vals))
    if k - 1 < len(vals):
        return bool(float(vals[k - 1]) > tau)
    return False


def block_smith_invariants(delta: np.ndarray, block_size: int = 40000) -> list[int]:
    """Return Smith normal form invariants using column blocks of ``delta``.

    Parameters
    ----------
    delta:
        Integer incidence matrix of the sheaf.
    block_size:
        Number of columns processed per block.

    Notes
    -----
    The matrix is split to avoid large dense factorizations. Each block
    is reduced with :func:`sympy.matrices.normalforms.smith_normal_form` and
    invariants are combined via greatest common divisors following
    a Mayer-Vietoris argument.
    """

    try:  # pragma: no cover - optional dependency
        import sympy as sp
        from sympy.matrices.normalforms import smith_normal_form
    except Exception as exc:  # pragma: no cover - sympy missing
        raise RuntimeError("sympy required for block_smith") from exc

    _, n = delta.shape
    if n == 0:
        return []

    invariants: list[int] = []
    for start in range(0, n, block_size):
        sub = sp.Matrix(delta[:, start : start + block_size])
        D, _, _ = smith_normal_form(sub)
        diag = [int(D[i, i]) for i in range(min(D.shape))]
        if not invariants:
            invariants = diag
        else:
            # combine via gcd to approximate the full SNF
            length = max(len(invariants), len(diag))
            invariants.extend([1] * (length - len(invariants)))
            diag.extend([1] * (length - len(diag)))
            invariants = [int(np.gcd(a, b)) for a, b in zip(invariants, diag)]

    return invariants


def block_smith(
    delta: np.ndarray, block_size: int = 40000, lam_thresh: float | None = None
) -> int:
    r"""Return the :math:`H^1` rank from a block-Smith reduction on ``delta``.

    Parameters
    ----------
    delta:
        Integer incidence matrix ``\delta`` of the sheaf.
    block_size:
        Number of columns processed per block when computing the Smith
        normal form.
    lam_thresh:
        Spectral shortcut threshold. If ``None`` it is read from
        ``configs/default.yaml`` under ``sheaf.lam_thresh``.

    Returns
    -------
    int
        Estimated rank of :math:`H^1`, i.e. the number of zero invariants. If
        the spectral bound is exceeded, ``0`` is returned immediately.
    """

    Delta = delta.T @ delta
    if lam_thresh is None:
        from ..utils.config import load_config

        cfg = load_config()
        lam_thresh = float(cfg.get("sheaf", {}).get("lam_thresh", 1e-3))

    try:  # pragma: no cover - optional dependency
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh

        vals = eigsh(
            sp.csr_matrix(Delta),
            k=min(5, Delta.shape[0] - 1),
            which="SM",
            return_eigenvectors=False,
        )
    except Exception:  # fall back to dense eigendecomposition
        vals = np.linalg.eigvalsh(Delta)

    if any(float(v) > lam_thresh for v in np.sort(np.asarray(vals))[:5]):
        return 0

    inv = block_smith_invariants(delta.astype(int), block_size=block_size)
    return sum(1 for i in inv if i == 0)


def validate_section(graph: nx.Graph, nodes: Iterable) -> float:
    """Return sheaf consistency score for the induced section."""

    sub = graph.subgraph(nodes)
    return sheaf_consistency_score(sub)
