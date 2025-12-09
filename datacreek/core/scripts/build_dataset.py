import argparse
import sys
from pathlib import Path

import networkx as nx

from datacreek.analysis.autotune import AutoTuneState
from datacreek.analysis.monitoring import push_metrics_gateway, start_metrics_server
from datacreek.core.dataset import DatasetBuilder
from datacreek.pipelines import DatasetType
from datacreek.utils.config import load_config


def run_pipeline(source: str, config: str, out: str, *, dry_run: bool = False) -> None:
    """Run the standard dataset pipeline from ingestion to export.

    Parameters
    ----------
    source:
        Path to the input document.
    config:
        YAML configuration file.
    out:
        Marker file written on successful completion when not in ``dry_run``.
    dry_run:
        If ``True``, skip persistence steps and metric push.
    """
    cfg = load_config(Path(config))
    start_metrics_server(int(cfg.get("monitor", {}).get("port", 8000)))
    ds = DatasetBuilder(DatasetType.QA, name=Path(source).stem)

    # 1 ingest -> atomes/molécules
    ds.ingest_file(source, config=cfg.get("ingest", {}))

    # 3 cleanup -> WCC, triangles, similarity, LP
    ds.gds_quality_check()

    # 4 tpl_validate -> persistance & Wasserstein (fix si besoin)
    ds.tpl_correct_graph(
        ds.graph.graph, epsilon=float(cfg.get("tpl", {}).get("eps_w1", 0.05))
    )

    # 5 fractal_dim -> d_B + σ
    ds.colour_box_dimension([1])

    # 6 embeddings -> N2V, GraphWave(Chebyshev), Poincaré
    ds.compute_graph_embeddings()
    ds.compute_graphwave_embeddings(scales=[1.0])
    ds.compute_poincare_embeddings()

    # 7 fusion_multiview → product + A-CCA + InfoNCE
    ds.compute_product_manifold_embeddings()
    ds.compute_aligned_cca_embeddings()

    # 8 index_ann → FAISS/HNSW build
    ds.graph.index.build()

    # 9 autotune → update θ
    state = AutoTuneState(
        tau=cfg.get("cleanup", {}).get("tau", 5),
        beta=cfg.get("ib", {}).get("beta", 0.01),
        eps=cfg.get("tpl", {}).get("eps_w1", 0.05),
        delta=0.05,
        p=cfg.get("embeddings", {}).get("node2vec", {}).get("p", 1.0),
        q=cfg.get("embeddings", {}).get("node2vec", {}).get("q", 1.0),
        dim=cfg.get("embeddings", {}).get("node2vec", {}).get("d", 64),
        alpha=cfg.get("embeddings", {}).get("alpha", 0.6),
        gamma=cfg.get("search", {}).get("gamma", 0.6),
        eta=cfg.get("search", {}).get("eta", 0.3),
    )
    labels = {n: 0 for n in ds.graph.graph.nodes()}
    penalty_cfg = cfg.get("penalty", {})
    ds.autotune_step(labels, [], state, penalty_cfg=penalty_cfg)

    # 10 generate_llm → prompts + sheaf_check
    ds.run_generation_layer(lambda x: x, query="auto", k=1)
    if ds.graph.sheaf_consistency_score() < 0.8:
        sys.exit(1)

    # 11 compress_cache → FractalNet + Mapper cache
    ds.prune_embeddings()

    ds.log_cycle_metrics()
    metrics = {
        "sigma_db": float(ds.graph.graph.get("fractal_sigma", 0.0)),
        "recall10": float(ds.graph.graph.get("recall10", 0.0)),
        "sheaf_score": float(ds.graph.sheaf_consistency_score()),
        "gw_entropy": float(ds.graph.graph.get("gw_entropy", 0.0)),
        "tpl_w1": float(ds.graph.graph.get("tpl_w1", 0.0)),
        "j_cost": float(ds.graph.graph.get("j_cost", 0.0)),
    }
    if not dry_run:
        push_metrics_gateway(metrics)
        ds.mark_exported()
        ds.save_neo4j()
        Path(out).write_text("completed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run datacreek dataset pipeline")
    parser.add_argument("--source", required=True, help="path to file to ingest")
    parser.add_argument("--config", required=True, help="config YAML path")
    parser.add_argument("--output", required=True, help="output marker file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run pipeline without saving graph or pushing metrics",
    )
    args = parser.parse_args()
    run_pipeline(args.source, args.config, args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
