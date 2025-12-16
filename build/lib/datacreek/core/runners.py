import logging
from dataclasses import dataclass
from typing import Iterable

from ..utils.config import load_config
from .knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass
class Node2VecRunner:
    """Run Node2Vec embeddings using configuration parameters."""

    graph: KnowledgeGraph

    def run(
        self,
        *,
        dimensions: int | None = None,
        walk_length: int = 10,
        num_walks: int = 50,
        workers: int = 1,
        seed: int = 0,
    ) -> None:
        cfg = load_config()
        emb_cfg = cfg.get("embeddings", {}).get("node2vec", {})
        dim_val = int(dimensions or emb_cfg.get("d", 128))
        p_val = float(emb_cfg.get("p", 1.0))
        q_val = float(emb_cfg.get("q", 1.0))
        self.graph.compute_node2vec_embeddings(
            dimensions=dim_val,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=workers,
            seed=seed,
            p=p_val,
            q=q_val,
        )
        try:
            import numpy as np

            embs = [
                np.asarray(self.graph.graph.nodes[n]["embedding"], dtype=float)
                for n in self.graph.graph.nodes
                if "embedding" in self.graph.graph.nodes[n]
            ]
            if embs:
                norms = np.linalg.norm(np.vstack(embs), axis=1)
                var_norm = float(np.var(norms))
                self.graph.graph.graph["var_norm"] = var_norm
                logger.info("var_norm=%.4f", var_norm)
                try:
                    from datacreek.analysis.monitoring import update_metric

                    update_metric("n2v_var_norm", var_norm)
                except Exception:  # pragma: no cover - optional Prometheus
                    pass
        except Exception:  # pragma: no cover - optional numpy
            pass


@dataclass
class GraphWaveRunner:
    """Run GraphWave embeddings with Chebyshev approximation."""

    graph: KnowledgeGraph

    def run(
        self,
        scales: Iterable[float] | None = None,
        *,
        num_points: int = 10,
        order: int = 7,
        dynamic: bool = False,
        gpu: bool = False,
    ) -> None:
        import networkx as nx

        from ..analysis.fractal import graphwave_embedding_chebyshev as _gwc
        from ..analysis.graphwave_bandwidth import update_graphwave_bandwidth

        G = self.graph.graph
        if scales is None or dynamic:
            t = update_graphwave_bandwidth(G)
            scales = [t]
        comps = list(nx.connected_components(G.to_undirected()))
        if len(comps) == 1:
            self.graph.compute_graphwave_embeddings(
                scales=scales,
                num_points=num_points,
                chebyshev_order=order,
                gpu=gpu,
            )
        else:
            for comp in comps:
                if len(comp) <= 1000:
                    continue
                sub = G.subgraph(comp)
                emb = _gwc(sub, scales, num_points=num_points, order=order)
                for node, vec in emb.items():
                    self.graph.graph.nodes[node]["graphwave_embedding"] = vec.tolist()
        gw_entropy = self.graph.graphwave_entropy()
        self.graph.graph.graph["gw_entropy"] = gw_entropy
        logger.info("gw_entropy=%.4f", gw_entropy)
        try:
            from datacreek.analysis.monitoring import gw_entropy as _gw_entropy_gauge
            from datacreek.analysis.monitoring import update_metric

            update_metric("gw_entropy", gw_entropy)
            if _gw_entropy_gauge is not None:
                _gw_entropy_gauge.set(float(gw_entropy))
        except Exception:  # pragma: no cover - optional Prometheus
            pass


@dataclass
class PoincareRunner:
    """Generate Poincar\xe9 embeddings and detect crowding."""

    graph: KnowledgeGraph

    def run(
        self,
        *,
        dim: int = 2,
        negative: int = 5,
        epochs: int = 50,
        learning_rate: float = 0.1,
        burn_in: int = 10,
    ) -> None:
        import numpy as np

        from ..analysis.fractal import poincare_embedding

        emb = poincare_embedding(
            self.graph.graph.to_undirected(),
            dim=dim,
            negative=negative,
            epochs=epochs,
            learning_rate=learning_rate,
            burn_in=burn_in,
        )
        for node, vec in emb.items():
            self.graph.graph.nodes[node]["poincare_embedding"] = vec.tolist()
        norms = [
            np.linalg.norm(self.graph.graph.nodes[n]["poincare_embedding"], ord=2)
            for n in self.graph.graph.nodes
            if "poincare_embedding" in self.graph.graph.nodes[n]
        ]
        radius_mean = float(np.mean(norms)) if norms else 0.0
        if radius_mean > 0.9:
            for n in self.graph.graph.nodes:
                if "poincare_embedding" not in self.graph.graph.nodes[n]:
                    continue
                v = np.asarray(
                    self.graph.graph.nodes[n]["poincare_embedding"], dtype=float
                )
                norm = np.linalg.norm(v) + 1e-8
                self.graph.graph.nodes[n]["poincare_embedding"] = (0.8 / norm) * v
            logger.info("crowding detected r_mean=%.3f, rescaled", radius_mean)
