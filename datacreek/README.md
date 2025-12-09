# Datacreek — README.td (scope: dossier `/datacreek` uniquement)

> **Objet** : document de référence exhaustif pour le **dossier `datacreek/`** (et seulement lui). Vous y trouverez : l’architecture, le rôle précis de chaque composant, leurs interactions, l’API exposée, ainsi qu’une **présentation mathématique de bout en bout** (hypergraphes, sheaves, TDA, spectre, hyperbolique, ANN…).

---

## Sommaire

* [Vue d’ensemble et objectifs](#vue-densemble-et-objectifs)
* [Architecture logique globale](#architecture-logique-globale)
* [Composants principaux (par répertoire/fichier)](#composants-principaux-par-répertoirefichier)

  * [`analysis/` — algorithmes, maths & gouvernance](#analysis--algorithmes-maths--gouvernance)
  * [`core/` — graphe de connaissances & pipeline dataset](#core--graphe-de-connaissances--pipeline-dataset)
  * [`models/` — clients & services LLM](#models--clients--services-llm)
  * [`routers/` & `api.py` — API FastAPI](#routers----apipy--api-fastapi)
  * [`server/` — UI Flask (lié aux tâches du pipeline)](#server--ui-flask-li%C3%A9-aux-t%C3%A2ches-du-pipeline)
  * [`backend/` & `backends.py` — abstraction CPU/GPU et backends](#backend----backendspy--abstraction-cpugpu-et-backends)
  * [`utils/` — utilitaires (chunking, extraction, cache…)](#utils--utilitaires-chunking-extraction-cache)
  * [`security/` & `dp/` — confidentialité différentielle & budget](#security----dp--confidentialit%C3%A9-diff%C3%A9rentielle--budget)
  * Autres : [`tasks.py`, `pipelines.py`, `schemas.py`, `db.py`, `config/`, `config_models.py`](#autres--taskspy-pipelinespy-schemaspy-dbpy-config-config_modelspy)
* [Flux de données — du fichier à la réponse](#flux-de-donn%C3%A9es--du-fichier-%C3%A0-la-r%C3%A9ponse)
* [Mathématiques de bout en bout](#math%C3%A9matiques-de-bout-en-bout)

  * [1) Graphe, hypergraphe & sheaf](#1-graphe-hypergraphe--sheaf)
  * [2) Topologie (TDA) & Mapper](#2-topologie-tda--mapper)
  * [3) Spectral : GraphWave, noyau de chaleur, Chebyshev/Hutch++](#3-spectral--graphwave-noyau-de-chaleur-chebyshevhutch)
  * [4) Géométrie hyperbolique (Poincaré)](#4-g%C3%A9om%C3%A9trie-hyperbolique-poincar%C3%A9)
  * [5) ANN & hybrid search (HNSW, IVFPQ) + auto‐tuning](#5-ann--hybrid-search-hnsw-ivfpq--auto-tuning)
  * [6) Multivues & gouvernance](#6-multivues--gouvernance)
* [APIs publiques (FastAPI)](#apis-publiques-fastapi)
* [Configuration & dépendances optionnelles](#configuration--d%C3%A9pendances-optionnelles)
* [Observabilité & qualité](#observabilit%C3%A9--qualit%C3%A9)
* [Bonnes pratiques d’exploitation](#bonnes-pratiques-dexploitation)

---

## Vue d’ensemble et objectifs

Le dossier **`datacreek/`** implémente une pile **RAG++** où le **retrieval** repose sur :

* un **graphe de connaissances** riche (documents, sections, chunks, entités, faits, hyper-arêtes) ;
* des **maths de graphe avancées** (hypergraphes, **Laplacien de sheaf**, **TDA**, **analyses spectrales**/ondelettes, **espace hyperbolique**) ;
* des **index ANN** (FAISS/HNSW/IVFPQ) et une **recherche hybride** (lexicale + sémantique) ;
* une **gouvernance des embeddings** (contrastes multivues, biais d’échelle, entropies, drift) ;
* une **intégration LLM** flexible (APIs compatibles OpenAI/vLLM) pour génération/curation.

Objectif : **améliorer la pertinence et la robustesse** du RAG au-delà du cosine sur embeddings, grâce à l’**information structurelle** et à des **contrôles mathématiques** de qualité (cohérence de sheaf, topologie persistante, spectre…).

---

## Architecture logique globale

```
                                ┌────────────────────────────────────────┐
 Ingestion (core/ingest.py)      │  Datasets (core/dataset_*.py)         │
  ├─ parsers/ & utils/           │   ├─ History/Events/Policy            │
  └─ chunking, extraction        │   ├─ Pipelines (pipelines.py)         │
                                 │   └─ LLMService (models/)             │
                                 └──────────────┬────────────────────────┘
                                                │
                                       ┌────────▼─────────┐
 Knowledge Graph (core/knowledge_graph.py)      │ nodes/edges + attrs
  ├─ NX DiGraph + (option) Neo4j/GDS            │
  ├─ hyperedges, simplex, entités/faits         │
  └─ features: Node2Vec/GraphWave/Poincaré      │
                                       └────────┬─────────┘
                                                │
                                         Analysis (analysis/)
  ┌─────────────────────────────────────────────────────────────────────┐
  │  hypergraph_conv.py  sheaf.py   tda_vectorize.py  graphwave_*.py    │
  │  chebyshev_diag.py   multiview.py   hyp_embed.py  hybrid_ann.py     │
  └─────────────────────────────────────────────────────────────────────┘
                                                │
                                    ANN index & Hybrid Search
                                  (analysis/index.py, hybrid_ann.py)
                                                │
                                   Retrieval → LLM (models/, core/create.py)
                                                │
                                         Curation/QA (core/curate.py)
                                                │
                                     API & UI (api.py, routers/, server/)
```

---

## Composants principaux (par répertoire/fichier)

### `analysis/` — algorithmes, maths & gouvernance

* **`hypergraph_conv.py`** :

  * *Laplacien d’hypergraphe* à partir d’une **matrice d’incidence** (B) pondérée ;
  * **convolutions spectrales** par **polynômes de Chebyshev** (ordre (K), normalisation par rayon spectral estimé) ;
  * *cache du rayon spectral* (power iteration) pour éviter des recomputations.
* **`hypergraph.py`** : embeddings **Hyper‑SAGNN‑like** sur hyperarêtes (attention sur nœuds de l’hyperarête, agrégation contextuelle).
* **`sheaf.py`** :

  * **Laplacien de sheaf** pour graphes signés (attribut d’arête `sheaf_sign`) ;
  * **incidence de sheaf** et **convolution** sheaf (mise à jour de features par (I - \alpha L_{sheaf})).
* **`tda_vectorize.py`**, **`tda.py`** :

  * **diagrammes de persistance** (Rips via GUDHI), *images de persistance* ;
  * signatures minhash de persistance pour *sketching* léger.
* **`mapper.py`**, **`tpl.py`** (et variantes) : **Mapper** (couverts/lens), **Topological Perception Layer** (ajout/suppression d’arêtes pour rapprocher la topologie cible : minimisation de distances de persistance).
* **`graphwave_bandwidth.py`**, **`graphwave_cuda.py`** :

  * estimation de **(\lambda_{max})** (puissance) et **mise à l’échelle** (t \approx c/\lambda_{max}) ;
  * **GraphWave** (noyau de chaleur, ondelettes) CPU/GPU (via `backend/array_api.py`).
* **`chebyshev_diag.py`** : estimation **trace diagonale du noyau de chaleur** (\mathrm{diag}(e^{-tL})) par **Chebyshev + Hutch++** (Bessel modifiées (I_v)), sparse/ dense.
* **`hyp_embed.py`** : embeddings **Poincaré** (MLP → projection boule), avec régularisation par **dimension fractale**.
* **`multiview.py`** :

  * **alignement multivues** (Euclidien Node2Vec ↔ Spectral GraphWave ↔ Hyperbolique) ;
  * entraînement sur **variété produit** et métriques d’accord.
* **`hybrid_ann.py`**, **`index.py`** : HNSW/IVFPQ (**FAISS**) + **nprobe** multi‑cellules, fallback dynamique vers HNSW si latence dépasse seuil ; suivi du **rappel**.
* **`governance.py`**, **`drift.py`**, **`monitoring.py`** :

  * biais d’échelle (**Wasserstein** des normes), **rayon hyperbolique moyen**, entropies ;
  * détection de drift (EWMA, seuils) ; exposition **Prometheus** (compteurs/jauges).
* **`fractal.py`** : utilitaires fractals (dimension, entropies), wrappers temps‑limite avec métriques.

> Ces briques **maths/algos** sont consommées par `core/knowledge_graph.py` et les pipelines.

---

### `core/` — graphe de connaissances & pipeline dataset

* **`knowledge_graph.py`** (**cœur**)

  * Graphe **NetworkX** (orienté) avec typage de nœuds (`document`, `section`, `chunk`, `entity`, `fact`, `image`, `audio`, etc.) et **hyper‑arêtes**.
  * Méthodes clés (extraits) :

    * construction : `add_document`, `add_section`, `add_chunk`, `add_entity`, `add_fact`, `add_hyperedge`, `add_simplex` ;
    * nettoyage & liens : `link_similar_*`, `deduplicate_chunks`, `mark_conflicting_facts`, `validate_coherence`, `prune_sources` ;
    * **recherche** : `search`, `search_hybrid`, `similar_by_hybrid`, `ann_hybrid_search`, `recall_at_k` ;
    * **explicabilité** : `explain_node`, `path_to_text`, `subgraph_to_text` ;
    * **maths intégrées** : `sheaf_laplacian`, `sheaf_consistency_score(_batched)`, `graph_entropy`, `subgraph_fractal_dimension`, `graph_information_bottleneck`, `spectral_bound_exceeded` ;
    * **gestion ANN** : `cypher_ann_query` (lors de l’usage Neo4j/GDS), index locaux (FAISS) ;
    * **multi‑vues** : stockage/alignement des embeddings (Node2Vec/GraphWave/Poincaré), métriques de gouvernance.
  * **Persistance/scale (optionnel)** : intégration **Neo4j Fabric/GDS** (cf. `neo4j_fabric.py`) pour calculs Node2Vec/similarités à l’échelle et multiplexage par *tenant*.
  * **Contrôles qualité** : calculs (\Delta\lambda) (variations spectrales), listing d’arêtes incohérentes (sheaf) exposé à l’UI de curation (`edge_review.py`).
* **`dataset_full.py` / `dataset_light.py` & `dataset.py`** :

  * Builder de **datasets** : historique, événements, **policy**, états (stage), gestion *light/full* (variable `DATACREEK_LIGHT_DATASET`).
  * Orchestration : génération, curation, ingestion, export, chargement Neo4j, etc., enchaînés via `pipelines.py` et `tasks.py`.
* **`ingest.py`** : validation des chemins, détection modalité (texte, audio, image), **chunking**, extraction (utils), et alimentation du graphe.
* **`create.py`** : génération **LLM** (CoT/QA/summaries), choix du modèle (OpenAI‑compatible/vLLM), formatage (`models/`).
* **`curate.py`** : curation/filtrage par LLM, ré‑annotation et mesures de qualité (peut s’appuyer sur le graphe pour contexte et *hard negatives*).
* **`context.py`, `cleanup.py`, `save_as.py`, `runners.py`** : glue et helpers (contexte appli, nettoyage, conversions, exécuteurs…).

---

### `models/` — clients & services LLM

* **`llm_client.py`** : client **multi‑provider** (OpenAI‑compatible, vLLM, endpoints HTTP), synchro/async, batchs, gestion des clefs et profils.
* **`llm_service.py`** : façade pratique synchronisée/async, transforme des *prompts* en appels du client.
* Autres : `content_type.py`, `qa.py` (structure QA), `results.py`, `stage.py`, `task_status.py`, `export_format.py`.

---

### `routers/` & `api.py` — API FastAPI

* **`api.py`** : application **FastAPI** principale (CORS, middleware **DPBudgetMiddleware**, routes importées). Expose gestion des datasets (création, génération, export…), sources, utilisateurs (selon présence de `db.py`).
* **`routers/vector_router.py`** : POST `/vector/search` → **recherche hybride** (lexicale + embedding) dans un dataset (auth via `X-API-Key`).
* **`routers/explain_router.py`** : exploration publique du graphe (auth) : sous‑graphe, **scores d’attention**, SVG encodé ; endpoints comme `/sheaf_diff` (liste d’arêtes incohérentes d’après le **Laplacien de sheaf**).

---

### `server/` — UI Flask (lié aux tâches du pipeline)

* **`server/app.py`** : application **Flask** rendant des vues (datasets, ingestion, curation, export) et invoquant les **tasks** (`tasks.py`) ; templates Jinja dans `server/templates/` (création, curation, login…).

> Même si l’API publique est FastAPI, l’UI d’admin/démo est fournie via Flask et pilote les tâches de pipeline.

---

### `backend/` & `backends.py` — abstraction CPU/GPU et backends

* **`backend/array_api.py`** : sélection dynamique **NumPy/CuPy** via `get_xp()` pour écrire du code **agnostique CPU/GPU**.
* **`backends.py`** : factories **Redis**, **Neo4j**, **S3** (boto3), avec surcharge par ENV/config (`utils/config.py`).

---

### `utils/` — utilitaires (chunking, extraction, cache…)

* **Texte & chunking** : `text.py`, `chunking.py` (fenêtres, *semantic split* via TF‑IDF), `format_converter.py`.
* **Extraction** : `entity_extraction.py`, `fact_extraction.py`, `modality.py`, `audio_*`, `whisper_batch.py`.
* **Récupération** : `retrieval.py` (TF‑IDF + HNSW, NearestNeighbors), `llm_processing.py`.
* **Cache & système** : `cache.py`, `redis_*`, `checksum.py`, `progress.py`, `backpressure.py`, `rate_limit.py`, `neo4j_breaker.py`.
* **Curation & QA** : `curation_agent.py`, `self_instruct.py`, `toolformer.py`, `emotion.py`, `quality_metrics.py`.

---

### `security/` & `dp/` — confidentialité différentielle & budget

* **`security/dp_budget.py`** : **sliding window** de budget (\epsilon) (24 h par défaut), prune, consommation, tests d’acceptation.
* **`security/dp_middleware.py`** : middleware FastAPI pour appliquer un budget par utilisateur/tenant.
* **`security/tenant_privacy.py`** : helpers multi‑tenant.
* **`dp/accountant.py`** : **Renyi DP moments accountant** (Mironov 2017) pour composer les (\epsilon) ; helpers `allow_request`.

---

### Autres : `tasks.py`, `pipelines.py`, `schemas.py`, `db.py`, `config/`, `config_models.py`

* **`tasks.py`** : *sugar* Celery‑like (retours `delay`/`apply_async`) + tâches dataset (ingest, extract_facts, generate, export, cleanup, load_neo4j…).
* **`pipelines.py`** : types de dataset, étapes (`PipelineStep`), coordination ingestion→index→génération→curation→export.
* **`schemas.py`** : **Pydantic** (ou fallback léger) — contraintes `DatasetName`, schémas d’entrées sorties API.
* **`db.py`** : modèles **SQLAlchemy** (User, Dataset, SourceData…), `SessionLocal`, initialisation ; intégrations Flask‑Login (fallback inclus si absent).
* **`config/` & `utils/config.py` & `config_models.py`** : chargement YAML (fallback si `yaml` absent), schéma (`ConfigSchema`), profils LLM (OpenAI/vLLM), GPU options.

---

## Flux de données — du fichier à la réponse

1. **Ingestion (`core/ingest.py`)** : validations, détection de modalité, **chunking** (fenêtres/TF‑IDF), normalisation, extraction (entités/faits), création de nœuds & arêtes dans le **KnowledgeGraph**.
2. **Indexation (`analysis/index.py`, `hybrid_ann.py`)** :

   * calcul/stockage d’**embeddings** multi‑vues (Node2Vec, GraphWave, Poincaré) ;
   * construction **ANN** (HNSW/IVFPQ) et **recherche hybride**.
3. **Contrôles structurels (`analysis/sheaf.py`, `tda_*`, `graphwave_*`)** :

   * **cohérence de sheaf**, **distances de persistance**, **entropies spectrales**, détection d’arêtes douteuses ((\Delta\lambda)).
4. **Retrieval (`core/knowledge_graph.py`)** : `search_hybrid`/`similar_by_hybrid` mettent en avant les candidats pertinents, pondérés par **structure** et **ANN**.
5. **Génération (`core/create.py`, `models/`)** : contexte consolidé (chemins, hyperarêtes, explicabilité), prompts envoyés via `LLMService`.
6. **Curation (`core/curate.py`)** : filtres qualité, ré‑annotation, *hard negatives* ; mise à jour du graphe (scores, liens, flags).
7. **Observation & réparation** : `edge_review.py` (UI), `/explain/sheaf_diff` (API), Prometheus (`analysis/monitoring.py`).

---

## Mathématiques de bout en bout

### 1) Graphe, hypergraphe & sheaf

* **Hypergraphe** : matrice d’incidence (B\in\mathbb{R}^{n\times m}) (n nœuds, m hyperarêtes), poids (w). **Laplacien** :
  [\Delta = B,\mathrm{diag}(w),B^\top.]
  Convolutions **spectrales** par **polynômes de Chebyshev** : pour un filtre (g) approché par (\sum_{k=0}^K \theta_k T_k(\tilde{\Delta})), avec (\tilde{\Delta}=\tfrac{2}{\lambda_{max}}\Delta - I) et récurrence (T_{k+1}(x)=2xT_k(x)-T_{k-1}(x)). Implémentation : `analysis/hypergraph_conv.py`.
* **Sheaf Laplacian** : graphe signé (attribut d’arête `sheaf_sign` ∈ {−1,+1}). **Incidence de sheaf** (B_s) (orientation pondérée par le signe), **Laplacien** (L_{sheaf}=B_s B_s^\top). **Convolution** : (X\leftarrow X - \alpha L_{sheaf} X). Implémentation : `analysis/sheaf.py` ; utilisé par `core/knowledge_graph.py` (scores de cohérence, diffs).
* **Hyper‑SAGNN** : pour une hyperarête (e) sur (p) nœuds avec features (X_e\in\mathbb{R}^{p\times d}), on applique **q/k/v** → scores d’attention → contexte → embedding moyen. Implémentation : `analysis/hypergraph.py` (+ version GPU stream : `hyper_sagnn_cuda.py`).

### 2) Topologie (TDA) & Mapper

* **Persistance (Rips)** : à partir de points/embeddings, calcul du diagramme ({(b_i,d_i)}) ; vecteur par **image de persistance** ou **signature minhash**. Implémentations : `analysis/tda_vectorize.py`, `tda.py`.
* **Mapper** : couverture par **lenses** (UMAP/trustworthiness guidée), agrégation par clusters → graphe Mapper ; métriques de **silhouette** pour la qualité. Impl.: `analysis/mapper.py`.
* **TPL (Topological Perception Layer)** : opérateur structurel qui **modifie** le graphe (ajout/suppression d’arêtes) pour **réduire la distance de persistance** vers une topologie cible (stabilité topologique). Impl.: `analysis/tpl*.py`.

### 3) Spectral : GraphWave, noyau de chaleur, Chebyshev/Hutch++

* **GraphWave** : signatures locales via **noyau de chaleur** (e^{-tL}), temps **auto‑ajusté** par (\lambda_{max}) estimé (puissance). Impl.: `analysis/graphwave_bandwidth.py`, `graphwave_cuda.py`.
* **Trace diagonale** (\mathrm{diag}(e^{-tL})) : approximation par **polynômes de Chebyshev** + **Hutch++** (améliore la variance du trace estimator), Bessel (I_v) pour coefficients. Impl.: `analysis/chebyshev_diag.py`.
* **Entropie/Dimension spectrale** : dérivées à partir de la **heat trace** et des spectres locaux (cf. `analysis/fractal.py`).

### 4) Géométrie hyperbolique (Poincaré)

* **Projection Poincaré** : MLP (\mathbb{R}^d\to\mathbb{D}^k), projection (x\mapsto x/\sqrt{1+\lVert x\rVert^2}) pour rester dans la boule. **Régularisation** par **dimension fractale** (boîte) : rapproche la complexité géométrique visée. Impl.: `analysis/hyp_embed.py` + recentrage barycentrique (`poincare_recentering.py`).
* **Navigation** : distances hyperboliques (arctanh des normes), *greedy* sur le graphe pour chemins informatifs (impl. associées dans `analysis/fractal.py` et wrappers KG).

### 5) ANN & hybrid search (HNSW, IVFPQ) + auto‐tuning

* **Indices** : HNSW (graphes proximaux) & **IVFPQ** (quantification produit, cellules). **`analysis/index.py`** gère construction, recherche, *fallback* dynamique vers HNSW si latence > seuil.
* **NProbe multi‑cellules** : `hybrid_ann.py` calcule (\texttt{nprobe_multi}\approx\sqrt{\texttt{n_cells}/\texttt{tables}}). Chargement CPU/GPU paramétré.
* **Rappel/latence** : `recall_at_k`, `ann_latency` (Prometheus), adaptation & **auto‑tuning** global via `analysis/autotune.py` (coût combinant rappel, entropies, IB, couverture fractale…).
* **Hybride** : combinaison **lexicale** (TF‑IDF/NN/HNSW) + **embedding** + **structure** (scores de chemin, hyperarêtes, ondelettes) dans `core/knowledge_graph.py` (`search_hybrid`, `hybrid_score`).

### 6) Multivues & gouvernance

* **Alignement** : `analysis/multiview.py` — corrèle Euclidien (Node2Vec), Spectral (GraphWave) et Hyperbolique (Poincaré) par pertes contrastives/produit de variétés.
* **Biais d’échelle** : `governance.py` — distances **Wasserstein** des distributions de normes (entre espaces), **rayon hyperbolique moyen**.
* **Drift** : `analysis/drift.py` & `security/dp_budget.py`/`dp/accountant.py` (côté budget), seuils d’alerte, EWMA, exposition Prometheus via `monitoring.py`.

---

## APIs publiques (FastAPI)

* **`POST /vector/search`** (`routers/vector_router.py`) : recherche hybride sur un dataset ; entrée = `dataset`, `query`, `top_k`, etc. Auth : `X-API-Key`.
* **`GET /explain/sheaf_diff`** (`routers/explain_router.py`) : renvoie des **arêtes incohérentes (sheaf)** ; **`GET /explain/...`** peut retourner un sous‑graphe + **scores d’attention** + SVG encodé.
* **`api.py`** regroupe ces routes et expose d’autres actions sur `Dataset`/`SourceData`/`User` si `db.py` est présent.

---

## Configuration & dépendances optionnelles

* **Configuration** : `config/config.yaml`, `utils/config.py`, `config/schema.py` (modèle `ConfigSchema`), `config_models.py` (profils OpenAI/vLLM, paramètres de génération, formats…). Variables ENV priment.
* **Dépendances optionnelles** (gérées par *try/except* avec **fallbacks**) : `faiss`, `cupy`, `scipy`, `gudhi`, `neo4j`, `redis`, `sklearn`, `prometheus_client`, `opentelemetry`, etc. Les fonctionnalités avancées s’activent si la lib est disponible, sinon un chemin **léger** est utilisé (ou l’option est désactivée).
* **GPU** : `backend/array_api.py` choisit automatiquement **CuPy** si présent et GPU détecté, sinon **NumPy**.

---

## Observabilité & qualité

* **Prometheus** : compteurs/jauges dans `analysis/monitoring.py` (ex. `embedding_cpu_seconds_total`, `ann_latency`, entropies, coûts auto‑tuning…).
* **Curation humaine** : `edge_review.py` (FastAPI) — liste des arêtes avec fort (\Delta\lambda), **accept/reject** ; journalisation en mémoire + écriture Neo4j (si configuré).
* **Explicabilité** : `explain_router.py` — sous‑graphes commentés, attention & SVG.
* **Politiques** : `core/dataset_*` — `Policy`, `HistoryEvent` ; **DP budget** (middleware + accountant Renyi) pour limiter l’usage.

---

## Bonnes pratiques d’exploitation

* **Profils d’installation** : démarrer **léger** (ANN + features de base), activer ensuite `faiss`, `cupy`, `gudhi`, `neo4j` selon besoins.
* **Index & métriques** : surveiller **rappel@k** et **latence** (p95) ; si latence ↑, activer **fallback HNSW** (déjà automatisé) ou réduire `k` ; analyser **biais d’échelle** et **rayon hyperbolique**.
* **Qualité structurelle** : examiner régulièrement `/explain/sheaf_diff` et **edge review UI** ; utiliser **TPL** pour corriger la topologie ; re‑échantillonner des embeddings **GraphWave** si (\lambda_{max}) a varié.
* **Coûts LLM** : tirer parti du retrieval **multivues/structurel** pour réduire tokens et modèles chers ; s’appuyer sur `LLMService` pour router vers des profils différents (vLLM local vs. API).

---

### Références de fichiers (non exhaustif, ciblé)

* **Maths** : `analysis/hypergraph_conv.py`, `analysis/hypergraph.py`, `analysis/sheaf.py`, `analysis/tda_vectorize.py`, `analysis/mapper.py`, `analysis/tpl*.py`, `analysis/graphwave_*.py`, `analysis/chebyshev_diag.py`, `analysis/hyp_embed.py`, `analysis/multiview.py`, `analysis/autotune.py`, `analysis/index.py`, `analysis/hybrid_ann.py`, `analysis/governance.py`, `analysis/drift.py`, `analysis/fractal.py`.
* **Graphe** : `core/knowledge_graph.py`, `neo4j_fabric.py`, `edge_review.py`.
* **Pipelines** : `core/dataset_*.py`, `core/ingest.py`, `core/create.py`, `core/curate.py`, `pipelines.py`, `tasks.py`.
* **LLM** : `models/llm_client.py`, `models/llm_service.py`.
* **API** : `api.py`, `routers/vector_router.py`, `routers/explain_router.py`.
* **Backends** : `backend/array_api.py`, `backends.py`.
* **Utils** : `utils/chunking.py`, `utils/retrieval.py`, `utils/entity_extraction.py`, `utils/fact_extraction.py`, `utils/cache.py`, `utils/neo4j_breaker.py`.
* **Sécurité** : `security/dp_budget.py`, `dp/accountant.py`, `security/dp_middleware.py`.

---

> **Scope rappel** : ce README ne couvre **que** le répertoire `datacreek/`. Les éléments hors de ce dossier (p. ex. `serving/`, `training/`, etc.) sont exclus par conception.
