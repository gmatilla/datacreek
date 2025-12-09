# Stable Interface Specification

This document enumerates the APIs considered stable for Datacreek v1.1. Any
future optimisation must preserve their behaviour and signature.

## Topology Pipeline
- `tpl_incremental(graph: nx.Graph, radius: int = 1, dimension: int = 1) -> dict`
  - Updates the persistence diagrams around each node if its neighbourhood hash
    changed. Returns a mapping of node IDs to diagrams.
  - The concatenation of updated diagrams is stored in `graph.graph["tpl_global"]`.

## Chebyshev GraphWave
- `chebyshev_diag_hutchpp(L: array, t: float, order: int = 7, samples: int = 64, rng=None) -> ndarray`
  - Estimates `diag(exp(-t L))` via Hutch++ with a Chebyshev polynomial approximation.
    Falls back to a Hutchinson estimator when SciPy is unavailable.
  - For sparse Laplacians, matrix-vector products may be offloaded to cuSPARSE.

## Autotune + Metrics
- `autotune_node2vec(kg, queries, ground_truth, k=10, var_threshold=1e-4, max_evals=40, max_minutes=30)`
  - Bayesian optimisation of Node2Vec parameters `p` and `q`. The search stops early when the embedding norm variance drops below `var_threshold`, when no improvement occurs for five iterations, or once `max_minutes` wall time has elapsed. The best `(p, q)` pair and a dataset hash are saved in `benchmarks/node2vec_last.json`.
- `autotune_nprobe(index, queries, truths, recall_goal=0.92)`
  - Tunes the FAISS `nprobe` parameter for IVFPQ indices using recall@100.

## Governance CI
- Continuous integration runs `bench_all.py` and fails when performance
  regresses by more than 5 % compared to `benchmarks/baseline.json`.
- GPU dependent tests are marked `faiss_gpu` and executed in a separate job.
  The helper `scripts/install_faiss.sh` installs `faiss-gpu` when CUDA is
  available, otherwise falling back to the CPU build.

## Monitoring
- Prometheus exporter provides `lmdb_evictions_total{cause="ttl"|"quota"|"manual"}`
  and `lmdb_eviction_last_ts{cause="ttl"|"quota"|"manual"}` counters. The label
  cardinality is purposely low (three causes) to keep metrics lightweight.
- The TTL manager adjusts `l1_ttl` via a PID controller configured by
  `cache.ttl_pid.target_hit_ratio`, `Kp`, `Ki` and `I_max`.
- Grafana dashboard `cache_overview.json` correlates eviction rates, Redis hit ratio
  and ingestion queue saturation. Alert `CachePressure` fires when the queue stays
  above 90 % while the hit ratio falls below the recorded threshold.

