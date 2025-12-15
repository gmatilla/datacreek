# Heavy dependency audit plan

The datacreek stack keeps a number of modules around optional tooling (`torch`, `faiss`, `gudhi`, `scipy`) and proprietary analyses (`graphwave_*`, `hyper_sagnn_*`, `tpl*`, etc.). The AGENTS checklist uses this guide to track the outstanding manual audit (see item 10). The audit consists of:

1. Enabling the missing libraries via `pip install torch faiss-cpu gudhi scipy`.
2. Running `python scripts/verify_heavy_modules.py` to confirm every target module can be imported.
3. Exercising that module manually (integration/unit tests, sample scripts, data ingestion) while noting any issues in this doc or in `docs/ledger.md`.

## Modules to inspect

| Module | Notes |
| --- | --- |
| `datacreek.analysis.graphwave_bandwidth` | GraphWave bandwidth evaluation. |
| `datacreek.analysis.graphwave_cuda` | CUDA-accelerated GraphWave utilities. |
| `datacreek.analysis.hyper_sagnn_cuda` | Hyper-SAGNN GPU embeddings. |
| `datacreek.analysis.chebyshev_diag` | Chebyshev diagonalization heuristics. |
| `datacreek.analysis.mapper` | Mapper-based embedding builder. |
| `datacreek.analysis.tda_fast` | Topological data analysis helper. |
| `datacreek.analysis.tpl*` | Persistent homology helpers (`tpl`, `tpl_incremental`, etc.). |
| `datacreek.analysis.information` | Information geometry scoring. |
| `datacreek.analysis.multiview` | Multiview feature extraction. |
| `datacreek.analysis.symmetry` | Symmetry detection utilities. |
| `datacreek.analysis.compression` | Compression heuristics for graphs. |
| `datacreek.analysis.governance` | Governance-related workflows. |
| `datacreek.analysis.rollback` | Rollback and recovery helpers. |
| `datacreek.analysis.ingestion` | Ingestion orchestration logic. |
| `datacreek.analysis.privacy` | Privacy metrics and logging. |
| `datacreek.analysis.explain_viz` | Explanation visualizations.

Other modules may be added to this list as the audit matures. When all imports succeed, capture any runtime failures in `docs/ledger.md` or a dedicated GitHub issue, then check off AGENTS item 10.
