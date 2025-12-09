from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from .fractal import bottleneck_distance, fractal_level_coverage
from .information import (
    graph_information_bottleneck,
    mdl_description_length,
    structural_entropy,
)
from .monitoring import update_metric
from .multiview import hybrid_score


@dataclass
class AutoTuneState:
    """State of the autotuning loop.

    Attributes
    ----------
    tau:
        Triangle count threshold for :func:`structural_entropy`.
    beta:
        Weight of the information bottleneck regularizer.
    eps:
        Allowed additive error for :func:`bottleneck_distance`.
    delta:
        MDL tolerance used when selecting motifs.
    p:
        Return bias of the Node2Vec random walk.
    q:
        In-out bias of the Node2Vec random walk.
    dim:
        Dimension of the Node2Vec embeddings.
    prev_graph:
        Previous graph snapshot for distance computations.
    """

    tau: int = 1
    beta: float = 0.1
    eps: float = 0.05
    delta: float = 0.05
    p: float = 1.0
    q: float = 1.0
    dim: int = 64
    alpha: float = 0.5
    gamma: float = 0.5
    eta: float = 0.25
    prev_graph: Optional[nx.Graph] = None
    coverage_min: float = 0.0
    prev_costs: list[float] = field(default_factory=list)
    jitter: float = 0.0
    likelihood: Any | None = None
    stagnation: int = 0


def recall_at_k(
    graph: nx.Graph,
    queries: Sequence[object],
    ground_truth: Dict[object, Sequence[object]],
    *,
    k: int = 10,
    gamma: float = 0.5,
    eta: float = 0.25,
) -> float:
    """Compute mean recall@k using the hybrid similarity score.

    Parameters
    ----------
    graph:
        Knowledge graph containing embeddings on nodes.
    queries:
        Nodes for which to compute retrieval performance.
    ground_truth:
        Mapping of query node to relevant nodes.
    k:
        Cutoff rank.
    gamma, eta:
        Weights used by :func:`hybrid_score`.
    """

    total = 0.0
    n = 0
    for q in queries:
        rel = set(ground_truth.get(q, []))
        if not rel:
            continue

        node_data = graph.nodes[q]
        n2v_q = node_data.get("embedding")
        gw_q = node_data.get("graphwave_embedding")
        hyp_q = node_data.get("poincare_embedding")
        if n2v_q is None or gw_q is None or hyp_q is None:
            continue

        scores = []
        for u, data in graph.nodes(data=True):
            if u == q:
                continue
            n2v_u = data.get("embedding")
            gw_u = data.get("graphwave_embedding")
            hyp_u = data.get("poincare_embedding")
            if n2v_u is None or gw_u is None or hyp_u is None:
                continue
            s = hybrid_score(
                n2v_u, n2v_q, gw_u, gw_q, hyp_u, hyp_q, gamma=gamma, eta=eta
            )
            scores.append((u, s))

        scores.sort(key=lambda x: x[1], reverse=True)
        retrieved = [u for u, _ in scores[:k]]
        hits = len(rel.intersection(retrieved))
        total += hits / len(rel)
        n += 1

    return total / n if n else 0.0


def kw_gradient(
    f, x: float, h: float = 1.0, n: int = 4
) -> float:  # pragma: no cover - stochastic
    """Return Kiefer-Wolfowitz stochastic gradient estimate.

    Parameters
    ----------
    f:
        Function taking a float and returning the scalar objective.
    x:
        Current parameter value.
    h:
        Perturbation size.
    n:
        Number of random evaluations.
    """
    rng = np.random.default_rng()
    grad = 0.0
    for _ in range(n):
        s = rng.choice([-1.0, 1.0])
        grad += s * f(x + s * h)
    return grad / (n * h)


def autotune_step(  # pragma: no cover - complex heuristic with heavy deps
    graph: nx.Graph,
    embeddings: Dict[object, Iterable[float]],
    labels: Dict[object, int],
    motifs: Iterable[nx.Graph],
    state: AutoTuneState,
    *,
    weights: Tuple[float, float, float, float, float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ),
    recall_data: Optional[
        Tuple[Sequence[object], Dict[object, Sequence[object]]]
    ] = None,
    penalty_cfg: Optional[Dict[str, float]] = None,
    k: int = 10,
    lr: float = 0.1,
    latency: float = 0.0,
) -> Dict[str, Any]:
    """Perform one step of the autotuning procedure.

    The function measures four objectives on ``graph`` using ``state``:
    structural entropy ``H`` (after filtering triangle counts),
    bottleneck distance ``D`` to the previous graph, information bottleneck
    loss ``I`` on ``embeddings``/``labels`` and MDL value ``M`` over ``motifs``.
    A weighted cost ``J`` is computed and ``state`` is updated via simple
    gradient heuristics on ``tau`` and ``eps``.

    Parameters
    ----------
    graph:
        Current knowledge graph snapshot.
    embeddings:
        Mapping of node to embedding vector used by the information bottleneck.
    labels:
        Mapping of node to integer class label.
    motifs:
        Candidate subgraphs used for the MDL score.
    state:
        Mutable autotuning state.
    weights:
        Weights ``(w1, w2, w3, w4, w5, w_sigma, w_cov, w_rec)`` of the
        multi-objective cost.
    recall_data:
        Optional tuple ``(queries, ground_truth)`` for computing recall@k with
        the hybrid similarity score.
    penalty_cfg:
        Optional mapping with keys ``lambda_sigma``, ``lambda_cov`` and
        ``w_rec`` controlling additional penalty terms.
    k:
        Rank cutoff used in the recall metric.
    lr:
        Learning rate for the gradient heuristics.
    latency:
        Search latency (seconds) used for the FAISS penalty term.

    Returns
    -------
    dict
        Dictionary with measured metrics and updated parameters.
    """

    H = structural_entropy(graph, state.tau)
    coverage = fractal_level_coverage(graph)
    if state.prev_graph is not None:
        D = bottleneck_distance(state.prev_graph, graph, approx_epsilon=state.eps)
    else:
        D = 0.0
    I = graph_information_bottleneck(embeddings, labels, beta=state.beta)
    M = mdl_description_length(graph, motifs, delta=state.delta)

    norms = [np.linalg.norm(v) for v in embeddings.values()] if embeddings else [0.0]
    var_phi = float(np.var(norms))

    recall = 0.0
    if recall_data is not None:
        queries, gt = recall_data
        recall = recall_at_k(
            graph,
            queries,
            gt,
            k=k,
            gamma=state.gamma,
            eta=state.eta,
        )

    sigma_db = float(graph.graph.get("fractal_sigma", 0.0))
    lambda_sigma = weights[5]
    lambda_cov = weights[6]
    w_rec = weights[7]
    w_lat = 0.0
    if penalty_cfg is not None:
        lambda_sigma = float(penalty_cfg.get("lambda_sigma", lambda_sigma))
        lambda_cov = float(penalty_cfg.get("lambda_cov", lambda_cov))
        w_rec = float(penalty_cfg.get("w_rec", w_rec))
        w_lat = float(penalty_cfg.get("w_lat", w_lat))
    J = (
        weights[0] * (-H)
        + weights[1] * D
        + weights[2] * I
        + weights[3] * M
        + weights[4] * (-var_phi)
    )
    J += lambda_sigma * max(0.0, sigma_db - 0.02)
    J += lambda_cov * max(0.0, state.coverage_min - coverage)
    J += w_rec * max(0.0, 0.9 - recall)
    J += w_lat * max(0.0, latency - 0.1)

    if state.prev_costs and abs(J - state.prev_costs[-1]) < 1e-3:
        state.stagnation += 1
    else:
        state.stagnation = 0
    state.prev_costs.append(J)
    update_metric("autotune_cost", float(J))
    try:
        from .monitoring import autotune_cost_g as _autotune_gauge

        if _autotune_gauge is not None:
            _autotune_gauge.set(float(J))
    except Exception:
        pass
    restart_gp = False
    if state.stagnation >= 5:
        if state.likelihood is not None and hasattr(state.likelihood, "noise"):
            state.likelihood.noise += 1e-3
        restart_gp = True
        state.stagnation = 0
        try:
            from .monitoring import gp_jitter_restarts_total as _gp_ctr

            if _gp_ctr is not None:
                _gp_ctr.inc()
        except Exception:
            pass

    # stochastic KW gradients for tau and eps
    grad_tau = kw_gradient(
        lambda t: structural_entropy(graph, int(max(1, t))), state.tau
    )
    state.tau = max(1, int(state.tau - lr * grad_tau))

    if state.prev_graph is not None:
        grad_eps = kw_gradient(
            lambda e: bottleneck_distance(
                state.prev_graph, graph, approx_epsilon=max(0.0, e)
            ),
            state.eps,
            h=0.01,
        )
        state.eps = max(0.0, state.eps + lr * grad_eps)

    I_next = graph_information_bottleneck(embeddings, labels, beta=state.beta + 0.01)
    grad_beta = (I_next - I) / 0.01
    state.beta = max(0.0, state.beta - lr * grad_beta)

    M_next = mdl_description_length(graph, motifs, delta=state.delta + 0.01)
    grad_delta = (M_next - M) / 0.01
    state.delta = max(0.0, state.delta - lr * grad_delta)

    state.prev_graph = graph.copy()

    if coverage < state.coverage_min:
        # slightly loosen the triangle threshold when coverage is too low
        state.tau += 1

    return {
        "entropy": H,
        "bottleneck": D,
        "ib_loss": I,
        "mdl": M,
        "coverage": coverage,
        "var_phi": var_phi,
        "recall": recall,
        "cost": J,
        "jitter": state.jitter,
        "tau": state.tau,
        "restart_gp": restart_gp,
        "eps": state.eps,
        "beta": state.beta,
        "delta": state.delta,
    }


def update_theta(state: AutoTuneState, metrics: Dict[str, float]) -> None:
    """Update ``state`` with new parameters and record the autotuning cost.

    Parameters
    ----------
    state:
        Mutable autotuning state to update.
    metrics:
        Dictionary returned by :func:`autotune_step`.
    """

    from .monitoring import j_cost

    J = float(metrics.get("cost", 0.0))
    if j_cost is not None:
        try:
            j_cost.set(J)
        except Exception:
            pass
    state.tau = int(metrics.get("tau", state.tau))
    state.eps = float(metrics.get("eps", state.eps))
    state.beta = float(metrics.get("beta", state.beta))
    state.delta = float(metrics.get("delta", state.delta))
    state.jitter = float(metrics.get("jitter", state.jitter))


def svgp_ei_propose(  # pragma: no cover - requires sklearn & scipy
    params: Sequence[Sequence[float]],
    scores: Sequence[float],
    bounds: Sequence[tuple[float, float]],
    *,
    m: int = 100,
    n_samples: int = 256,
) -> np.ndarray:
    """Return new parameter vector maximizing Expected Improvement.

    A sparse Gaussian Process regression model is fitted on ``params`` and
    ``scores``. ``m`` subsamples are used as inducing points. ``n_samples``
    random candidates are drawn within ``bounds`` and the one with highest
    expected improvement over the best observed score is returned.
    """

    import logging
    from math import inf

    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.gaussian_process.kernels import ConstantKernel as C
    from sklearn.gaussian_process.kernels import Matern

    X = np.asarray(list(params), dtype=float)
    y = np.asarray(list(scores), dtype=float)
    if len(X) == 0:
        raise ValueError("no observations provided")
    idx = np.linspace(0, len(X) - 1, min(len(X), m), dtype=int)
    X_sub = X[idx]
    y_sub = y[idx]

    grads = []
    pairs = list(zip(X_sub, y_sub))
    for (p1, s1), (p2, s2) in zip(pairs[:-1], pairs[1:]):
        dist = np.linalg.norm(np.asarray(p2) - np.asarray(p1))
        if dist == 0:
            continue
        grads.append(abs(s2 - s1) / dist)
    var_grad = np.var(grads[-10:]) if len(grads) >= 2 else 0.0
    if var_grad > 0.5:
        kernel = C(1.0, (1e-3, 1e3)) * Matern(nu=1.5)
        logging.getLogger(__name__).info("dynamic_kernel=matern var=%.3f", var_grad)
    else:
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
    gp.fit(X_sub, y_sub)

    bounds = np.asarray(bounds, dtype=float)
    dim = bounds.shape[0]
    rng = np.random.default_rng()
    candidates = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, dim))

    y_best = y.min()
    best_x = None
    best_ei = -inf
    for x in candidates:
        mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
        sigma = float(sigma) + 1e-9
        improvement = y_best - mu
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        if ei > best_ei:
            best_ei = float(ei)
            best_x = x
    return best_x if best_x is not None else candidates[0]
