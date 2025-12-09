"""Datacreek application."""

import pkgutil
from typing import TYPE_CHECKING

__version__ = "0.0.2"

# Avoid heavy imports at module load time. Relevant classes and functions
# are available via submodules such as ``datacreek.pipelines`` or
# ``datacreek.core``.

__all__: list[str] = pkgutil.get_data(__name__, "_all_names.txt").decode().split()

if TYPE_CHECKING:  # pragma: no cover - used for type checking only
    from .analysis.fractal import (
        bottleneck_distance,
        box_counting_dimension,
        box_cover,
        build_fractal_hierarchy,
        build_mdl_hierarchy,
        fractal_information_density,
        fractalize_graph,
        fractalize_optimal,
        graph_fourier_transform,
        graph_lacunarity,
        graphwave_embedding,
        inverse_graph_fourier_transform,
        laplacian_energy,
        laplacian_spectrum,
        mdl_optimal_radius,
        minimize_bottleneck_distance,
        persistence_diagrams,
        persistence_entropy,
        poincare_embedding,
        spectral_density,
        spectral_dimension,
        spectral_entropy,
        spectral_gap,
    )
    from .analysis.generation import (
        generate_graph_rnn_like,
        generate_graph_rnn_sequential,
        generate_graph_rnn_stateful,
        generate_netgan_like,
    )
    from .analysis.hypergraph import hyper_sagnn_embeddings
    from .config_models import GenerationSettings
    from .core.dataset import DatasetBuilder
    from .core.ingest import ingest_into_dataset
    from .core.ingest import process_file as ingest_file
    from .core.ingest import to_kg
    from .core.knowledge_graph import KnowledgeGraph
    from .pipelines import (
        PIPELINES,
        DatasetType,
        GenerationPipeline,
        TrainingGoal,
        get_dataset_types_for_training,
        get_pipeline,
        get_pipelines_for_training,
        get_trainings_for_dataset,
        run_generation_pipeline,
        run_generation_pipeline_async,
    )
    from .utils.emotion import detect_emotion
    from .utils.fact_extraction import extract_facts
    from .utils.image_captioning import caption_image


def __getattr__(name: str):  # pragma: no cover - dynamic lazy imports
    if name == "DatasetBuilder":
        from .core.dataset import DatasetBuilder as _DB

        return _DB
    if name == "Atom":
        from .core.dataset import Atom as _Atom

        return _Atom
    if name in {"ingest_file", "to_kg", "ingest_into_dataset"}:
        from .core.ingest import ingest_into_dataset as _ingest_into_dataset
        from .core.ingest import process_file as _ingest_file
        from .core.ingest import to_kg as _to_kg

        return {
            "ingest_file": _ingest_file,
            "to_kg": _to_kg,
            "ingest_into_dataset": _ingest_into_dataset,
        }[name]
    if name == "KnowledgeGraph":
        from .core.knowledge_graph import KnowledgeGraph as _KG

        return _KG
    if name == "GenerationSettings":
        from .config_models import GenerationSettings as _GS

        return _GS
    if name in {
        "DatasetType",
        "GenerationPipeline",
        "TrainingGoal",
        "get_pipeline",
        "get_trainings_for_dataset",
        "get_dataset_types_for_training",
        "get_pipelines_for_training",
    }:
        from .pipelines import DatasetType as _DT
        from .pipelines import GenerationPipeline as _GP
        from .pipelines import TrainingGoal as _TG
        from .pipelines import get_dataset_types_for_training as _gdtft
        from .pipelines import get_pipeline as _gp
        from .pipelines import get_pipelines_for_training as _gpft
        from .pipelines import get_trainings_for_dataset as _gtfd

        mapping = {
            "DatasetType": _DT,
            "GenerationPipeline": _GP,
            "TrainingGoal": _TG,
            "get_pipeline": _gp,
            "get_trainings_for_dataset": _gtfd,
            "get_dataset_types_for_training": _gdtft,
            "get_pipelines_for_training": _gpft,
        }

        return mapping[name]
    if name == "extract_facts":
        from .utils.fact_extraction import extract_facts as _ef

        return _ef
    if name in {"run_generation_pipeline", "run_generation_pipeline_async"}:
        from .pipelines import run_generation_pipeline as _rgp
        from .pipelines import run_generation_pipeline_async as _rgp_async

        return {
            "run_generation_pipeline": _rgp,
            "run_generation_pipeline_async": _rgp_async,
        }[name]
    if name in {
        "curate_qa_pairs",
        "curate_qa_pairs_async",
        "filter_rated_pairs",
        "apply_curation_threshold",
    }:
        from .core.curate import apply_curation_threshold as _act
        from .core.curate import curate_qa_pairs as _cqp
        from .core.curate import curate_qa_pairs_async as _cqpa
        from .core.curate import filter_rated_pairs as _frp

        return {
            "curate_qa_pairs": _cqp,
            "curate_qa_pairs_async": _cqpa,
            "filter_rated_pairs": _frp,
            "apply_curation_threshold": _act,
        }[name]
    if name == "box_cover":
        from .analysis.fractal import box_cover as _bc

        return _bc
    if name == "box_counting_dimension":
        from .analysis.fractal import box_counting_dimension as _bcd

        return _bcd
    if name == "colour_box_dimension":
        from .analysis.fractal import colour_box_dimension as _cbd

        return _cbd
    if name == "persistence_entropy":
        from .analysis.fractal import persistence_entropy as _pe

        return _pe
    if name == "graphwave_embedding":
        from .analysis.fractal import graphwave_embedding as _ge

        return _ge
    if name == "graphwave_embedding_chebyshev":
        from .analysis.fractal import graphwave_embedding_chebyshev as _gec

        return _gec
    if name == "minimize_bottleneck_distance":
        from .analysis.fractal import minimize_bottleneck_distance as _mbd

        return _mbd
    if name == "bottleneck_distance":
        from .analysis.fractal import bottleneck_distance as _bd

        return _bd
    if name == "mdl_optimal_radius":
        from .analysis.fractal import mdl_optimal_radius as _mr

        return _mr
    if name == "persistence_diagrams":
        from .analysis.fractal import persistence_diagrams as _pd

        return _pd
    if name == "persistence_wasserstein_distance":
        from .analysis.fractal import persistence_wasserstein_distance as _pwd

        return _pwd
    if name == "spectral_dimension":
        from .analysis.fractal import spectral_dimension as _sd

        return _sd
    if name == "laplacian_spectrum":
        from .analysis.fractal import laplacian_spectrum as _ls

        return _ls
    if name == "spectral_entropy":
        from .analysis.fractal import spectral_entropy as _se

        return _se
    if name == "spectral_gap":
        from .analysis.fractal import spectral_gap as _sg

        return _sg
    if name == "laplacian_energy":
        from .analysis.fractal import laplacian_energy as _le

        return _le
    if name == "lacunarity":
        from .analysis.fractal import graph_lacunarity as _gl

        return _gl
    if name == "spectral_density":
        from .analysis.fractal import spectral_density as _sdn

        return _sdn
    if name == "graph_fourier_transform":
        from .analysis.fractal import graph_fourier_transform as _gft

        return _gft
    if name == "inverse_graph_fourier_transform":
        from .analysis.fractal import inverse_graph_fourier_transform as _igft

        return _igft
    if name == "graph_information_bottleneck":
        from .analysis.information import graph_information_bottleneck as _gib

        return _gib
    if name == "graph_entropy":
        from .analysis.information import graph_entropy as _ge

        return _ge
    if name == "subgraph_entropy":
        from .analysis.information import subgraph_entropy as _se

        return _se
    if name == "structural_entropy":
        from .analysis.information import structural_entropy as _str_e

        return _str_e
    if name == "prototype_subgraph":
        from .analysis.information import prototype_subgraph as _ps

        return _ps
    if name == "sheaf_laplacian":
        from .analysis.sheaf import sheaf_laplacian as _sl

        return _sl
    if name == "sheaf_convolution":
        from .analysis.sheaf import sheaf_convolution as _sc

        return _sc
    if name == "sheaf_neural_network":
        from .analysis.sheaf import sheaf_neural_network as _snn

        return _snn
    if name == "sheaf_first_cohomology":
        from .analysis.sheaf import sheaf_first_cohomology as _sfc

        return _sfc
    if name == "sheaf_first_cohomology_blocksmith":
        from .analysis.sheaf import sheaf_first_cohomology_blocksmith as _sfcbs

        return _sfcbs
    if name == "resolve_sheaf_obstruction":
        from .analysis.sheaf import resolve_sheaf_obstruction as _rso

        return _rso
    if name == "sheaf_consistency_score_batched":
        from .analysis.sheaf import sheaf_consistency_score_batched as _scsb

        return _scsb
    if name == "spectral_bound_exceeded":
        from .analysis.sheaf import spectral_bound_exceeded as _sbe

        return _sbe
    if name == "quality_check":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.quality_check
    if name == "gds_quality_check":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.gds_quality_check
    if name == "validate_topology":
        from .core.knowledge_graph import KnowledgeGraph as _KG

        return _KG.validate_topology
    if name == "optimize_topology_iterative":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.optimize_topology_iterative
    if name == "quality_layer":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.run_quality_layer
    if name == "fractal_layer":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.run_fractal_layer
    if name == "embedding_layer":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.run_embedding_layer
    if name == "hypergraph_layer":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.run_hypergraph_layer
    if name == "generation_layer":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.run_generation_layer
    if name == "generation_layer_async":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.run_generation_layer_async
    if name == "topological_perception_layer":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.run_topological_perception_layer
    if name == "tpl_correct_graph" or name == "tpl_incremental":
        from .core.dataset import DatasetBuilder

        return {
            "tpl_correct_graph": DatasetBuilder.tpl_correct_graph,
            "tpl_incremental": DatasetBuilder.tpl_incremental,
        }[name]
    if name == "compression_layer":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.run_compression_layer
    if name == "information_layer":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.run_information_layer
    if name == "topological_signature_hash":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.topological_signature_hash
    if name == "export_layer":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.run_export_layer
    if name == "orchestrator":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.run_orchestrator
    if name == "orchestrator_async":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.run_orchestrator_async
    if name == "poincare_embedding":
        from .analysis.fractal import poincare_embedding as _peb

        return _peb
    if name == "recenter_embeddings":
        from .analysis.poincare_recentering import recenter_embeddings as _re

        return _re
    if name == "compute_hyperbolic_hypergraph_embeddings":
        from .core.knowledge_graph import KnowledgeGraph as _KG

        return _KG.compute_hyperbolic_hypergraph_embeddings
    if name == "fractalize_graph":
        from .analysis.fractal import fractalize_graph as _fg

        return _fg
    if name == "fractalize_optimal":
        from .analysis.fractal import fractalize_optimal as _fo

        return _fo
    if name == "build_fractal_hierarchy":
        from .analysis.fractal import build_fractal_hierarchy as _bfh

        return _bfh
    if name == "build_mdl_hierarchy":
        from .analysis.fractal import build_mdl_hierarchy as _bmh

        return _bmh
    if name == "caption_image":
        from .utils.image_captioning import caption_image as _ci

        return _ci
    if name == "detect_emotion":
        from .utils.emotion import detect_emotion as _de

        return _de
    if name == "detect_modality":
        from .utils.modality import detect_modality as _dm

        return _dm
    if name == "md5_file":
        from .utils.checksum import md5_file as _md5

        return _md5
    if name == "partition_files_to_atoms":
        from .analysis.ingestion import partition_files_to_atoms as _pfa

        return _pfa
    if name == "transcribe_audio":
        from .analysis.ingestion import transcribe_audio as _ta

        return _ta
    if name == "transcribe_audio_batch":
        from .utils.whisper_batch import transcribe_audio_batch as _tab

        return _tab
    if name == "blip_caption_image":
        from .analysis.ingestion import blip_caption_image as _bci

        return _bci
    if name == "parse_code_to_atoms":
        from .analysis.ingestion import parse_code_to_atoms as _pca

        return _pca
    if name == "generate_graph_rnn_like":
        from .analysis.generation import generate_graph_rnn_like as _gg

        return _gg
    if name == "generate_graph_rnn_stateful":
        from .analysis.generation import generate_graph_rnn_stateful as _grs

        return _grs
    if name == "generate_graph_rnn_sequential":
        from .analysis.generation import generate_graph_rnn_sequential as _grs2

        return _grs2
    if name == "generate_netgan_like":
        from .analysis.generation import generate_netgan_like as _gn

        return _gn
    if name == "fractal_information_density":
        from .analysis.fractal import fractal_information_density as _fid

        return _fid
    if name == "fractal_coverage":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.fractal_coverage
    if name == "ensure_fractal_coverage":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.ensure_fractal_coverage
    if name == "diversification_score":
        from .analysis.fractal import diversification_score as _ds

        return _ds
    if name == "hyperbolic_neighbors":
        from .core.knowledge_graph import KnowledgeGraph as _KG

        return _KG.hyperbolic_neighbors
    if name == "hyperbolic_reasoning":
        from .core.knowledge_graph import KnowledgeGraph as _KG

        return _KG.hyperbolic_reasoning
    if name == "hyperbolic_hypergraph_reasoning":
        from .core.knowledge_graph import KnowledgeGraph as _KG

        return _KG.hyperbolic_hypergraph_reasoning
    if name == "hyperbolic_multi_curvature_reasoning":
        from .core.knowledge_graph import KnowledgeGraph as _KG

        return _KG.hyperbolic_multi_curvature_reasoning
    if name == "tpl_correct_graph" or name == "tpl_incremental":
        from .core.knowledge_graph import KnowledgeGraph as _KG

        return {
            "tpl_correct_graph": _KG.tpl_correct_graph,
            "tpl_incremental": _KG.tpl_incremental,
        }[name]
    if name == "compute_distmult_embeddings":
        from .core.knowledge_graph import KnowledgeGraph as _KG

        return _KG.compute_distmult_embeddings
    if name == "neighborhood_to_sentence":
        from .utils.graph_text import neighborhood_to_sentence as _nts

        return _nts
    if name == "subgraph_to_text":
        from .utils.graph_text import subgraph_to_text as _st

        return _st
    if name == "graph_to_text":
        from .utils.graph_text import graph_to_text as _gt

        return _gt
    if name == "betti_number":
        from .core.knowledge_graph import KnowledgeGraph as _KG

        return _KG.betti_number
    if name == "coverage_stats":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.coverage_stats
    if name == "invariants_dashboard":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.invariants_dashboard
    if name == "mapper_nerve":
        from .analysis.mapper import mapper_nerve as _mn

        return _mn
    if name == "inverse_mapper":
        from .analysis.mapper import inverse_mapper as _im

        return _im
    if name == "fractal_net_prune":
        from .analysis.fractal import fractal_net_prune as _fp

        return _fp
    if name == "prune_fractalnet":
        from .analysis.compression import prune_fractalnet as _pf

        return _pf
    if name == "fractalnet_compress":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.fractalnet_compress
    if name == "fractalnet_compress":
        from .analysis.fractal import fractalnet_compress as _fc

        return _fc
    if name == "graphwave_entropy":
        from .analysis.fractal import graphwave_entropy as _ge

        return _ge
    if name == "embedding_entropy":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.embedding_entropy
    if name == "graph_entropy":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.graph_entropy
    if name == "subgraph_entropy":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.subgraph_entropy
    if name == "structural_entropy":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.structural_entropy
    if name in {
        "product_embedding",
        "aligned_cca",
        "multiview_contrastive_loss",
        "meta_autoencoder",
        "alignment_correlation",
        "average_hyperbolic_radius",
        "scale_bias_wasserstein",
        "mitigate_bias_wasserstein",
        "filter_semantic_cycles",
        "entropy_triangle_threshold",
        "rollback_gremlin_diff",
        "SheafSLA",
        "governance_metrics",
        "k_out_randomized_response",
        "DPBudgetManager",
        "DPBudget",
    }:
        from .analysis import filtering as _flt
        from .analysis import governance as _g
        from .analysis import multiview as _mv
        from .analysis import privacy as _p
        from .security import dp_budget as _dp
        from .security import tenant_privacy as _tp

        if hasattr(_mv, name):
            return getattr(_mv, name)
        if hasattr(_g, name):
            return getattr(_g, name)
        if hasattr(_p, name):
            return getattr(_p, name)
        if hasattr(_flt, name):
            return getattr(_flt, name)
        if hasattr(_tp, name):
            return getattr(_tp, name)
        if hasattr(_dp, name):
            return getattr(_dp, name)
        if name == "rollback_gremlin_diff":
            from .analysis.rollback import rollback_gremlin_diff as _rgd

            return _rgd
        if name == "SheafSLA":
            from .analysis.rollback import SheafSLA as _sla

            return _sla
        return getattr(_dp, name)
    if name == "embedding_box_counting_dimension":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.embedding_box_counting_dimension
    if name == "colour_box_dimension":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.colour_box_dimension
    if name == "ensure_graphwave_entropy":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.ensure_graphwave_entropy
    if name == "hyper_sagnn_embeddings":
        from .analysis.hypergraph import hyper_sagnn_embeddings as _hs

        return _hs
    if name == "select_mdl_motifs":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.select_mdl_motifs
    if name == "mdl_description_length":
        from .analysis.information import mdl_description_length as _mdl

        return _mdl
    if name in {"AutoTuneState", "autotune_step", "kw_gradient", "autotune_nprobe"}:
        from .analysis.autotune import AutoTuneState as _AS
        from .analysis.autotune import kw_gradient as _kw
        from .analysis.nprobe_tuning import autotune_nprobe as _anp
        from .core.dataset import DatasetBuilder as _DB

        if name == "AutoTuneState":
            return _AS
        if name == "kw_gradient":
            return _kw
        if name == "autotune_nprobe":
            return _anp
        return _DB.autotune_step
    if name in {"export_embeddings_pg", "query_topk_pg"}:
        from .plugins.pgvector_export import export_embeddings_pg as _eep
        from .plugins.pgvector_export import query_topk_pg as _qt

        return {"export_embeddings_pg": _eep, "query_topk_pg": _qt}[name]
    if name in {"propose_merge_split", "record_feedback", "fine_tune_from_feedback"}:
        from .utils.curation_agent import fine_tune_from_feedback as _ft
        from .utils.curation_agent import propose_merge_split as _ps
        from .utils.curation_agent import record_feedback as _rf

        mapping = {
            "propose_merge_split": _ps,
            "record_feedback": _rf,
            "fine_tune_from_feedback": _ft,
        }

        return mapping[name]
    if name == "prune_embeddings":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.prune_embeddings
    if name in {
        "xor_encrypt",
        "xor_decrypt",
        "encrypt_pii_fields",
        "decrypt_pii_fields",
    }:
        from .utils.crypto import decrypt_pii_fields as _dp
        from .utils.crypto import encrypt_pii_fields as _ep
        from .utils.crypto import xor_decrypt as _xd
        from .utils.crypto import xor_encrypt as _xe

        return {
            "xor_encrypt": _xe,
            "xor_decrypt": _xd,
            "encrypt_pii_fields": _ep,
            "decrypt_pii_fields": _dp,
        }[name]
    if name == "ingestion_layer":
        from .core.dataset import DatasetBuilder

        return DatasetBuilder.run_ingestion_layer
    if name == "InvariantPolicy":
        from .core.dataset import InvariantPolicy as _IP

        return _IP
    if name == "monitor_after":
        from .core.dataset import monitor_after as _ma

        return _ma
    if name == "start_policy_monitor":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.start_policy_monitor
    if name == "stop_policy_monitor":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.stop_policy_monitor
    if name == "start_policy_monitor_thread":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.start_policy_monitor_thread
    if name == "stop_policy_monitor_thread":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.stop_policy_monitor_thread
    if name == "ingest_text_atoms":
        from .core.dataset import DatasetBuilder as _DB

        return _DB.ingest_text_atoms
    if name in {
        "detect_automorphisms",
        "quotient_by_symmetry",
        "automorphism_group_order",
    }:
        from .core.dataset import DatasetBuilder as _DB

        mapping = {
            "detect_automorphisms": _DB.detect_automorphisms,
            "automorphism_group_order": _DB.automorphism_group_order,
            "quotient_by_symmetry": _DB.quotient_by_symmetry,
        }

        return mapping[name]
    if name == "get_template" or name == "PromptTemplate" or name == "validate_output":
        from .templates.library import PromptTemplate as _PT
        from .templates.library import get_template as _gtmpl
        from .templates.library import validate_output as _vo

        return {"get_template": _gtmpl, "PromptTemplate": _PT, "validate_output": _vo}[
            name
        ]
    raise AttributeError(name)


import logging
import os

try:  # optional dependencies for GraphRNN
    from .analysis.fractal import ensure_graphrnn_checkpoint
except Exception:  # pragma: no cover - optional dependency missing

    def ensure_graphrnn_checkpoint(*_args, **_kwargs):
        return None


try:
    from .core.knowledge_graph import start_cleanup_watcher as _start_cleanup_watcher
except Exception:  # pragma: no cover - optional dependency missing

    def _start_cleanup_watcher(*_args, **_kwargs) -> None:
        return None


try:
    from .utils.config import ORIGINAL_CONFIG_PATH
except Exception:  # pragma: no cover - optional dependency missing
    ORIGINAL_CONFIG_PATH = None  # type: ignore[assignment]

if ORIGINAL_CONFIG_PATH:
    try:  # pragma: no cover - optional watcher startup
        _start_cleanup_watcher(os.getenv("DATACREEK_CONFIG", ORIGINAL_CONFIG_PATH))
    except Exception:  # pragma: no cover - watcher may fail
        pass

try:  # optional heavy dependency
    ensure_graphrnn_checkpoint()
except Exception:  # pragma: no cover - optional dependency missing
    pass
