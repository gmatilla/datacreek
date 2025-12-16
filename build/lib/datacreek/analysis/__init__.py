"""Utilities for graph analysis and fractal measures."""

__all__ = [
    "bottleneck_distance",
    "persistence_wasserstein_distance",
    "box_counting_dimension",
    "box_cover",
    "graphwave_embedding",
    "embedding_box_counting_dimension",
    "latent_box_dimension",
    "fractal_dim_embedding",
    "colour_box_dimension",
    "mdl_optimal_radius",
    "persistence_diagrams",
    "persistence_entropy",
    "fractalize_graph",
    "fractalize_optimal",
    "minimize_bottleneck_distance",
    "build_fractal_hierarchy",
    "build_mdl_hierarchy",
    "poincare_embedding",
    "learn_hyperbolic_projection",
    "recenter_embeddings",
    "spectral_dimension",
    "laplacian_spectrum",
    "spectral_entropy",
    "spectral_density",
    "graph_lacunarity",
    "graph_fourier_transform",
    "inverse_graph_fourier_transform",
    "graph_information_bottleneck",
    "graph_entropy",
    "subgraph_entropy",
    "structural_entropy",
    "prototype_subgraph",
    "sheaf_laplacian",
    "sheaf_convolution",
    "sheaf_neural_network",
    "sheaf_first_cohomology",
    "sheaf_first_cohomology_blocksmith",
    "sheaf_consistency_score_batched",
    "resolve_sheaf_obstruction",
    "sheaf_consistency_score",
    "fractal_information_density",
    "fractal_level_coverage",
    "diversification_score",
    "generate_graph_rnn_stateful",
    "generate_graph_rnn_sequential",
    "hyperbolic_reasoning",
    "hyperbolic_hypergraph_reasoning",
    "automorphisms",
    "automorphism_orbits",
    "automorphism_group_order",
    "quotient_graph",
    "mapper_nerve",
    "inverse_mapper",
    "mapper_full",
    "mapper_to_json",
    "autotune_mapper_overlap",
    "tune_overlap",
    "bootstrap_db",
    "bootstrap_sigma_db",
    "fractal_net_prune",
    "fractalnet_compress",
    "prune_fractalnet",
    "FractalNetPruner",
    "graphwave_entropy",
    "embedding_entropy",
    "hyper_sagnn_embeddings",
    "hyper_sagnn_head_drop_embeddings",
    "hyper_sagnn_embeddings_stream",
    "hyperedge_attention_scores",
    "fp8_quantize",
    "fp8_dequantize",
    "online_pca_reduce",
    "reduce_pca",
    "fractal_encoder",
    "mdl_description_length",
    "select_mdl_motifs",
    "product_embedding",
    "train_product_manifold",
    "aligned_cca",
    "cca_align",
    "load_cca",
    "hybrid_score",
    "multiview_contrastive_loss",
    "meta_autoencoder",
    "search_with_fallback",
    "recall10",
    "tpl_correct_graph",
    "tpl_incremental",
    "alignment_correlation",
    "average_hyperbolic_radius",
    "scale_bias_wasserstein",
    "governance_metrics",
    "AutoTuneState",
    "autotune_step",
    "svgp_ei_propose",
    "kw_gradient",
    "autotune_nprobe",
    "profile_nprobe",
    "autotune_node2vec",
    "spectral_bound_exceeded",
    "filter_semantic_cycles",
    "entropy_triangle_threshold",
    "rollback_gremlin_diff",
    "SheafSLA",
    "start_metrics_server",
    "push_metrics_gateway",
    "update_metric",
    "estimate_lambda_max",
    "update_graphwave_bandwidth",
    "chebyshev_diag_hutchpp",
    "search_hnsw_pq",
    "graphwave_embedding_gpu",
    "persistence_image",
    "persistence_landscape",
    "persistence_silhouette",
    "betti_curve",
    "diagram_entropy",
    "augment_embeddings_with_persistence",
    "hypergraph_laplacian",
    "chebyshev_conv",
    "chebyshev_heat_kernel_gpu",
    "fast_persistence_diagrams",
    "sheaf_hyper_bridge_score",
    "top_k_incoherent",
    "explain_to_svg",
    "baseline_mean",
    "embedding_mmd",
    "save_baseline",
    "load_baseline",
]


def __getattr__(name: str):  # pragma: no cover - dynamic lazy imports
    if name in {
        "bottleneck_distance",
        "box_counting_dimension",
        "box_cover",
        "fractalize_graph",
        "fractalize_optimal",
        "graph_fourier_transform",
        "graphwave_embedding",
        "inverse_graph_fourier_transform",
        "laplacian_spectrum",
        "mdl_optimal_radius",
        "minimize_bottleneck_distance",
        "build_fractal_hierarchy",
        "build_mdl_hierarchy",
        "persistence_diagrams",
        "persistence_entropy",
        "persistence_wasserstein_distance",
        "poincare_embedding",
        "spectral_density",
        "spectral_dimension",
        "spectral_entropy",
        "graph_lacunarity",
        "fractal_information_density",
        "fractal_level_coverage",
        "diversification_score",
        "hyperbolic_reasoning",
        "hyperbolic_hypergraph_reasoning",
        "fractal_net_prune",
        "fractalnet_compress",
        "prune_fractalnet",
        "FractalNetPruner",
        "graphwave_entropy",
        "embedding_entropy",
        "embedding_box_counting_dimension",
        "latent_box_dimension",
        "fractal_dim_embedding",
        "colour_box_dimension",
        "bootstrap_db",
        "bootstrap_sigma_db",
    }:
        from . import fractal as _f

        return getattr(_f, name)
    if name == "recenter_embeddings":
        from .poincare_recentering import recenter_embeddings as _re

        return _re
    if name == "FractalNetPruner":
        from .compression import FractalNetPruner as _fp

        return _fp
    if name in {"fp8_quantize", "fp8_dequantize"}:
        from .compression import fp8_dequantize as _dq
        from .compression import fp8_quantize as _q

        return {"fp8_quantize": _q, "fp8_dequantize": _dq}[name]
    if name in {"online_pca_reduce", "fractal_encoder"}:
        from .fractal_encoder import fractal_encoder as _fe
        from .fractal_encoder import online_pca_reduce as _opr

        return {"online_pca_reduce": _opr, "fractal_encoder": _fe}[name]
    if name in {
        "search_with_fallback",
        "recall10",
        "autotune_nprobe",
        "profile_nprobe",
        "autotune_node2vec",
    }:
        from . import index as _idx
        from .node2vec_tuning import autotune_node2vec as _an2
        from .nprobe_tuning import autotune_nprobe as _anp
        from .nprobe_tuning import profile_nprobe as _pn

        return {
            "search_with_fallback": _idx.search_with_fallback,
            "recall10": _idx.recall10,
            "autotune_nprobe": _anp,
            "profile_nprobe": _pn,
            "autotune_node2vec": _an2,
        }[name]
    if name in {
        "hyper_sagnn_embeddings",
        "hyper_sagnn_head_drop_embeddings",
        "hyperedge_attention_scores",
        "hyper_sagnn_embeddings_stream",
    }:
        from .hyper_sagnn_cuda import hyper_sagnn_embeddings_stream as _hs_stream
        from .hypergraph import hyper_sagnn_embeddings as _hs
        from .hypergraph import hyper_sagnn_head_drop_embeddings as _hd
        from .hypergraph import hyperedge_attention_scores as _att

        return {
            "hyper_sagnn_embeddings": _hs,
            "hyper_sagnn_head_drop_embeddings": _hd,
            "hyperedge_attention_scores": _att,
            "hyper_sagnn_embeddings_stream": _hs_stream,
        }[name]
    if name in {"mdl_description_length", "select_mdl_motifs"}:
        from .information import mdl_description_length as _mdl_desc
        from .information import select_mdl_motifs as _mdl_sel

        return _mdl_desc if name == "mdl_description_length" else _mdl_sel
    if name in {
        "product_embedding",
        "train_product_manifold",
        "aligned_cca",
        "cca_align",
        "load_cca",
        "hybrid_score",
        "multiview_contrastive_loss",
        "meta_autoencoder",
        "tpl_correct_graph",
        "alignment_correlation",
        "average_hyperbolic_radius",
        "scale_bias_wasserstein",
        "governance_metrics",
    }:
        from . import multiview as _mv

        if name == "cca_align" or hasattr(_mv, name):
            return getattr(_mv, name)
        from . import governance as _g

        return getattr(_g, name)
    if name == "tpl_correct_graph" or name == "tpl_incremental":
        from .tpl import tpl_correct_graph as _tcg
        from .tpl_incremental import tpl_incremental as _tpli

        return {"tpl_correct_graph": _tcg, "tpl_incremental": _tpli}[name]
    if name in {
        "automorphisms",
        "automorphism_orbits",
        "quotient_graph",
        "automorphism_group_order",
    }:
        from . import symmetry as _s

        return getattr(_s, name)
    if name in {
        "mapper_nerve",
        "inverse_mapper",
        "mapper_full",
        "mapper_to_json",
        "autotune_mapper_overlap",
    }:
        from . import mapper as _m

        return getattr(_m, name)
    if name == "graph_information_bottleneck":
        from .information import graph_information_bottleneck as _gib

        return _gib
    if name == "graph_entropy":
        from .information import graph_entropy as _ge

        return _ge
    if name == "subgraph_entropy":
        from .information import subgraph_entropy as _se

        return _se
    if name == "structural_entropy":
        from .information import structural_entropy as _str_e

        return _str_e
    if name == "prototype_subgraph":
        from .information import prototype_subgraph as _ps

        return _ps
    if name in {"AutoTuneState", "autotune_step", "svgp_ei_propose", "kw_gradient"}:
        from .autotune import AutoTuneState as _AS
        from .autotune import autotune_step as _at
        from .autotune import kw_gradient as _kw
        from .autotune import svgp_ei_propose as _sv

        return {
            "AutoTuneState": _AS,
            "autotune_step": _at,
            "svgp_ei_propose": _sv,
            "kw_gradient": _kw,
        }[name]
    if name == "sheaf_laplacian":
        from .sheaf import sheaf_laplacian as _sl

        return _sl
    if name == "sheaf_convolution":
        from .sheaf import sheaf_convolution as _sc

        return _sc
    if name == "sheaf_neural_network":
        from .sheaf import sheaf_neural_network as _snn

        return _snn
    if name == "sheaf_first_cohomology":
        from .sheaf import sheaf_first_cohomology as _sfc

        return _sfc
    if name == "sheaf_first_cohomology_blocksmith":
        from .sheaf import sheaf_first_cohomology_blocksmith as _sfcbs

        return _sfcbs
    if name == "resolve_sheaf_obstruction":
        from .sheaf import resolve_sheaf_obstruction as _rso

        return _rso
    if name == "sheaf_consistency_score":
        from .sheaf import sheaf_consistency_score as _scs

        return _scs
    if name == "sheaf_consistency_score_batched":
        from .sheaf import sheaf_consistency_score_batched as _scsb

        return _scsb
    if name == "spectral_bound_exceeded":
        from .sheaf import spectral_bound_exceeded as _sbe

        return _sbe
    if name == "generate_graph_rnn_stateful":
        from .generation import generate_graph_rnn_stateful as _grs

        return _grs
    if name == "generate_graph_rnn_sequential":
        from .generation import generate_graph_rnn_sequential as _grs2

        return _grs2
    if name == "filter_semantic_cycles":
        from .filtering import filter_semantic_cycles as _fsc

        return _fsc
    if name == "entropy_triangle_threshold":
        from .filtering import entropy_triangle_threshold as _ett

        return _ett
    if name == "rollback_gremlin_diff":
        from .rollback import rollback_gremlin_diff as _rgd

        return _rgd
    if name == "SheafSLA":
        from .rollback import SheafSLA as _sla

        return _sla
    if name in {"start_metrics_server", "push_metrics_gateway", "update_metric"}:
        from .monitoring import push_metrics_gateway as _pg
        from .monitoring import start_metrics_server as _sms
        from .monitoring import update_metric as _um

        return {
            "start_metrics_server": _sms,
            "push_metrics_gateway": _pg,
            "update_metric": _um,
        }[name]
    if name in {
        "estimate_lambda_max",
        "update_graphwave_bandwidth",
        "chebyshev_diag_hutchpp",
        "search_hnsw_pq",
        "graphwave_embedding_gpu",
        "persistence_image",
        "persistence_landscape",
        "persistence_silhouette",
        "betti_curve",
        "diagram_entropy",
        "augment_embeddings_with_persistence",
        "reduce_pca",
        "hypergraph_laplacian",
        "chebyshev_conv",
        "chebyshev_heat_kernel_gpu",
        "fast_persistence_diagrams",
        "sheaf_hyper_bridge_score",
        "top_k_incoherent",
    }:
        from .chebyshev_diag import chebyshev_diag_hutchpp as _cdh
        from .graphwave_bandwidth import estimate_lambda_max as _el
        from .graphwave_bandwidth import update_graphwave_bandwidth as _ugb
        from .graphwave_cuda import chebyshev_heat_kernel_gpu as _gwk
        from .graphwave_cuda import graphwave_embedding_gpu as _gwe
        from .hybrid_ann import search_hnsw_pq as _hpq
        from .hypergraph_conv import chebyshev_conv as _cc
        from .hypergraph_conv import hypergraph_laplacian as _hl
        from .sheaf_hyper_bridge import sheaf_hyper_bridge_score as _shb
        from .sheaf_hyper_bridge import top_k_incoherent as _tki
        from .tda_fast import fast_persistence_diagrams as _fpd
        from .tda_vectorize import augment_embeddings_with_persistence as _ap
        from .tda_vectorize import betti_curve as _bc
        from .tda_vectorize import diagram_entropy as _de
        from .tda_vectorize import persistence_image as _pi
        from .tda_vectorize import persistence_landscape as _pl
        from .tda_vectorize import persistence_silhouette as _ps
        from .tda_vectorize import reduce_pca as _rp

        return {
            "estimate_lambda_max": _el,
            "update_graphwave_bandwidth": _ugb,
            "chebyshev_diag_hutchpp": _cdh,
            "search_hnsw_pq": _hpq,
            "graphwave_embedding_gpu": _gwe,
            "persistence_image": _pi,
            "persistence_landscape": _pl,
            "persistence_silhouette": _ps,
            "betti_curve": _bc,
            "diagram_entropy": _de,
            "augment_embeddings_with_persistence": _ap,
            "reduce_pca": _rp,
            "hypergraph_laplacian": _hl,
            "chebyshev_conv": _cc,
            "chebyshev_heat_kernel_gpu": _gwk,
            "fast_persistence_diagrams": _fpd,
            "sheaf_hyper_bridge_score": _shb,
            "top_k_incoherent": _tki,
        }[name]
    if name in {"baseline_mean", "embedding_mmd", "save_baseline", "load_baseline"}:
        from .drift import baseline_mean as _bm
        from .drift import embedding_mmd as _em
        from .drift import load_baseline as _lb
        from .drift import save_baseline as _sb

        return {
            "baseline_mean": _bm,
            "embedding_mmd": _em,
            "save_baseline": _sb,
            "load_baseline": _lb,
        }[name]
    if name == "learn_hyperbolic_projection":
        from .hyp_embed import learn_hyperbolic_projection as _lh

        return _lh
    if name == "explain_to_svg":
        from .explain_viz import explain_to_svg as _ets

        return _ets
    raise AttributeError(name)
