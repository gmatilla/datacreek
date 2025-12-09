# Datacreek

![Complexity](https://img.shields.io/badge/average%20complexity-A-brightgreen)
[![Coverage Status](https://coveralls.io/repos/github/OWNER/REPO/badge.svg?branch=main)](https://coveralls.io/github/OWNER/REPO?branch=main)
![nightly](https://github.com/…/actions/workflows/nightly.yml/badge.svg)
![docstring quality](https://img.shields.io/badge/docstring--quality-0.80%2B-brightgreen)

Tool for generating high-quality synthetic datasets to fine-tune LLMs.

Generate reasoning traces and QA pairs and save them to common fine-tuning formats.

> [Checkout our guide on using the tool to unlock task-specific reasoning in Llama-3 family](https://github.com/meta-llama/datacreek/tree/main/use-cases/adding_reasoning_to_llama_3)

# What does Datacreek offer? 

Fine-Tuning Large Language Models is easy. There are many mature tools that you can use to fine-tune Llama model family using various post-training techniques.

### Why target data preparation?

Multiple tools support standardized formats. However, most of the time your dataset is not structured in "user", "assistant" threads or in a format that plays well with typical fine-tuning tools.

This toolkit simplifies the journey of:

- Using a LLM (vLLM or any local/external API endpoint) to generate examples
- Converting your existing files to fine-tuning friendly formats
- Parsing additional formats like images (via [unstructured](https://github.com/Unstructured-IO/unstructured)) and audio files (transcribed with SpeechRecognition + pocketsphinx). Extracted text from these sources is inserted in the knowledge graph while preserving the original file path
- `unstructured` is used to parse PDF, DOCX, PPTX, HTML and image files, ensuring consistent handling across formats
- Files are partitioned with `unstructured` so text and images become individual elements linked in the graph
- Text is cleaned with `unstructured` before being inserted into the knowledge graph
- Images are captioned with BLIP and stored with the caption as `alt_text`
- Audio files are transcribed via Whisper and linked back to the originating chunk
- Quantities are converted to SI units when `quantulum3` and `pint` are installed
- Mapper operations use a hierarchical cache. Results are stored in Redis and
  LMDB with an on-disk fallback. The LMDB layer enforces
  `cache.l2_max_mb` and purges old entries when this limit is exceeded,
  emitting an `[L2-EVICT]` log message with the number of removed keys.
- Optionally extracting named entities and standalone facts during ingestion
- Advanced chunking options including semantic, contextual and summarized splitting
- Creating synthetic datasets
- Extracting standalone facts into a knowledge graph
- Supporting various formats of post-training fine-tuning
- Asynchronous tasks are executed via Celery with status stored in Redis
- Datasets persist graphs in Redis and Neo4j and keep a version history
- Dataset progress and history endpoints expose current status and past versions via `/datasets/<name>/progress`, `/datasets/<name>/history` and `/datasets/<name>/versions`
- Neo4j indexes are created on startup to speed up queries
  (`NEO4J_INIT_INDEXES=0` disables this step)
- Datasets can be cloned to experiment with different cleaning steps
- Optional RedisGraph integration for fast local queries
- GraphWave embeddings can run on GPU via cuSPARSE when :mod:`cupy` is installed
- Mapper CLI supports `--lens graphwave_energy` with `--cover-resolution` and
  `--cover-overlap` to tune the cover parameters
- Hyperbolic training introduces losses `L_geo` and `L_frac` controlled by
  `--alpha` and `--beta` flags
- Continuous monitoring with an `InvariantPolicy` enforces entropy and
  spectral limits at each stage
- Configuration files are hot-reloaded by a watcher so threshold updates are
  applied without restarting
- Before each cleanup run `verify_thresholds()` checks that the in-memory
  values match the YAML configuration and raises an error on mismatch
- The watcher logs `[CFG-HOT] cleanup thresholds updated at <timestamp>` whenever the YAML values change
- Exports can optionally be uploaded to S3 for convenient remote storage
  when `S3_BUCKET`, `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are configured
- A policy monitor thread periodically enforces the `InvariantPolicy`
- An async orchestrator (`run_orchestrator_async()`) runs the full pipeline with
  concurrent LLM calls
- Graph maintenance utilities dynamically reconfigure fractal levels and add
  hypergraph relations via `run_hypergraph_layer()`
- A topological perception layer refines embeddings with
  `run_topological_perception_layer()`
- A semantic perception helper `apply_perception()` updates nodes and runs
  `node_similarity()` to detect duplicates. Use
  `apply_perception_all_nodes()` to transform the entire graph and validate
  uniqueness in bulk
  - Prompts may optionally encrypt PII fields when passed
    an `encrypt_key` to `export_prompts()`, ensuring author and
    organization metadata are not stored in plaintext
  - `export_prompts()` records the `topological_signature_hash` of
    the knowledge graph and tracks traversal counts.
    Call `coverage_stats()` to identify unexplored regions of the graph
    and target them in future generation runs.

## Fine-tune with Unsloth + TRL

Use the adaptive pipeline to train models with 4-bit loading, task detection,
curriculum scheduling and optional RLHF rewards. Refer to
[`docs/train_pipeline.md`](docs/train_pipeline.md) for a complete guide.

```bash
python -m cli.train --model Uns7B --dataset-path data.jsonl --task auto
```

## Workflow Overview

1. **Ingestion** – parse documents or URLs from multiple sources, clean the text and store everything in a knowledge graph.
2. **Knowledge graph operations** – deduplicate, resolve entities and link related chunks to improve data quality.
3. **Dataset generation** – choose a dataset type and training goal. The application selects the appropriate pipeline and formats the output accordingly.
4. **Initial curation** – a first filter removes low quality results.
5. **Dataset cleanup** – additional operations can be applied interactively to refine the dataset.
6. **Export** – download locally or upload to services like Hugging Face in the desired format.

| Stage | Options | Purpose |
|-------|---------|---------|
| **Ingestion** | Multiple file formats (PDF, DOCX, PPTX, TXT), web pages and YouTube URLs. Automatic cleaning and optional entity extraction. | Produce clean text chunks stored in the knowledge graph. |
| **Knowledge graph ops** | Helpers like `deduplicate_chunks`, `resolve_entities` and various linking utilities. | Improve quality and connectivity of the graph. |
| **Dataset generation** | Choose dataset type, training goal and output format. | The correct pipeline generates samples and formats them as requested. |
| **Curation** | Configure a quality threshold and batch size. | Drop obviously bad samples directly after generation. |
| **Dataset cleanup** | Reuse graph helper functions plus formatting utilities. | Finalize the dataset before exporting. |
| **Export** | Local download or HF upload in JSONL, Alpaca, ChatML or OpenAI FT style. | Deliver the dataset in the desired place and format. |

### Step Reference

Below is a quick overview of the main options and operations exposed at each stage.

**Ingestion**

- `extract_entities`, `extract_facts` – add entity and fact nodes while parsing
- `high_res`, `ocr` – improved PDF and image handling
- `chunk_method` – `basic`, `sliding`, `semantic`, `contextual` or `summary`

**Knowledge graph operations**

- `deduplicate_chunks()` – drop identical chunks
- `resolve_entities()` – merge aliases into canonical entities
- `link_*()` – connect nodes that mention the same entities
- `prune_sources([...])` – remove unwanted data from the graph
- `compute_graph_embeddings()` – build Node2Vec embeddings
- `compute_node2vec_gds(driver)` – run Neo4j GDS Node2Vec and store vectors
- `build_faiss_index()` – create a FAISS index from stored embeddings
- `scripts/ann_benchmark.py` – measure P95 latency and recall of ANN backends
- `scripts/bench_all.py` – aggregate CPU, GPU, memory usage and recall metrics, and compare against `benchmarks/baseline.json` to fail CI on >5% regression. The script also records ingestion throughput, GraphWave rate, Whisper realtime factor and PID convergence time.
- `scripts/install_faiss.sh` – helper installing `faiss-gpu` when CUDA is present or `faiss-cpu` otherwise.
- `mark_conflicting_facts()` – flag contradictory statements

**Dataset generation**

- `dataset_type` – qa, cot, kg, vqa, pref_pair…
- `training_goal` – SFT, DPO, PPO, etc.
- `fmt` – jsonl, alpaca, chatml, openai-ft

**Curation**

- `threshold` – minimum quality rating
- `batch_size` – number of pairs rated together
- `temperature` – sampling temperature for the rating prompt

**Dataset cleanup**

- `clean_chunks()` – normalize whitespace and remove markup
- `normalize_date_fields()` – standardize date attributes
- `deduplicate_pairs()` – remove near-duplicate examples

**Export**

- `fmt` – choose the output format
- `repo` – optionally push to a Hugging Face repo

### SaaS Pipeline Summary

Below is a concise walkthrough of the SaaS flow and which helpers implement each
step:

1. **Multimodal ingestion** – functions in `ingest.py` rely on
   `unstructured` to partition documents and on BLIP/Whisper to caption images
   and transcribe audio (`partition_pdf`, `partition_html`, `caption_image`,
   `transcribe_audio`).
2. **Atomic splitting** – `molecule_from_atoms()` groups contiguous elements and
   preserves relations such as `NEXT` or `CAPTION_OF`.
3. **Knowledge graph build** – `KnowledgeGraph.add_document()` and
   `add_chunk()` insert nodes and edges, while linking helpers (e.g.
   `link_chunks_by_entity`) establish semantics.
4. **Neo4j quality checks** – `DatasetBuilder.gds_quality_check()` runs
   `wcc`, `triangleCount` and `nodeSimilarity` to remove duplicates and weak
   links.
5. **Fractalization & embeddings** – `build_mdl_hierarchy()` performs box
   covering and `compute_graph_embeddings()` materializes Node2Vec vectors.
   `compute_node2vec_gds()` can leverage Neo4j GDS for large graphs and
   `build_faiss_index()` exposes fast cosine search.
6. **Automated monitoring** – `monitor_and_remediate()` runs after each
   major step to enforce entropy, fractal dimension and spectral gap
   thresholds.  Limits are defined by an `InvariantPolicy` dataclass and
   cleanup loops continue until metrics fall within range.
7. **Dynamic reconfiguration** – `dynamic_reconfigure()` refreshes fractal levels
   and resolves sheaf obstructions as the graph evolves.
8. **Hypergraph layer** – `run_hypergraph_layer()` computes Hyper‑SAGNN
   embeddings and inserts predicted hyperedges via
   `update_hypergraph_structure()`.
9. **Topological perception** – `run_topological_perception_layer()` adjusts the
   graph topology using sheaf constraints and updates fractal levels.
10. **Policy enforcement** – `_enforce_policy()` wraps dynamic reconfiguration,
   hypergraph suggestions and invariant monitoring so each pipeline stage
   remains within entropy, fractal and spectral limits.
11. **Automatic triggers** – dataset mutations decorated with
   `monitor_after()` enforce the policy right after each update.
12. **Background monitor** – `start_policy_monitor()` periodically applies the
   policy in an asynchronous loop until `stop_policy_monitor()` is called.
13. **Threaded monitoring** – `start_policy_monitor_thread()` runs the same
   loop in a separate thread when `auto_monitor=True`.
   ```python
   from datacreek import (
       DatasetBuilder,
       DatasetType,
       InvariantPolicy,
       start_policy_monitor_thread,
   )

   ds = DatasetBuilder(DatasetType.QA, name="demo", auto_monitor=True)
   # adjust thresholds if necessary
   ds.policy = InvariantPolicy(entropy_max=6.0, gap_min=0.1)
   start_policy_monitor_thread(ds, [1], interval=120.0)
   ```
14. **Async orchestration** – `run_orchestrator_async()` executes the full
   pipeline with concurrent LLM calls via `run_generation_layer_async`.
15. **Semantic perception** – `apply_perception()` modifies node text and now
   automatically triggers `node_similarity()` to flag near duplicates.
16. **Export & datasets** – `run_generation_pipeline()` produces QA pairs or
   other dataset types which are curated via `curate.py` and saved through
   `save_as.py`.
17. **Dataset progress** – monitor status via `/datasets/<name>/progress`,
   view past events via `/datasets/<name>/history` and manage stored
   versions through `/datasets/<name>/versions`.
18. **Remote upload** – exported datasets can be uploaded to S3 when
   `S3_BUCKET`, `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are set.

## Fractal metrics and confidence

Knowledge graph utilities provide MDL-guided box covering to assign `fractal_level` annotations. QA generation records a `confidence` score computed by searching up to three hops between subject and object.
Useful helpers:
- `build_mdl_hierarchy` returns successive coarse graphs until the description length increases.
- `annotate_mdl_levels` tags each node with its fractal level based on that hierarchy.
- `fact_confidence` searches up to three hops to rate the reliability of a statement.
- `DatasetBuilder.export_prompts(auto_fractal=True)` automatically annotates
  nodes with fractal levels using the MDL hierarchy when none are present.
- Generated QA pairs include a `confidence` field so downstream
  applications can score factual reliability directly.



# How does Datacreek offer it?

The toolkit exposes a REST API that mirrors the main data preparation steps. All
operations are asynchronous and keyed by users:

- `/tasks/ingest` for converting raw files to text
- `/tasks/generate` for creating datasets
- `/tasks/curate` for quality filtering
- `/tasks/save` for exporting in common fine-tuning formats
- `/datasets` to manage generated datasets

 All behaviour is driven from a YAML configuration file that you can override with your own values.



### Installation

Create a dedicated environment and install the project dependencies:

```bash
conda create -n synthetic-data python=3.10
conda activate synthetic-data
git clone https://github.com/meta-llama/datacreek.git
cd datacreek
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional helpers such as quantity normalization during text cleanup require
additional dependencies:

```bash
pip install quantulum3 pint
```

### CI setup

Use `scripts/install_faiss.sh` to install FAISS depending on GPU availability.
Run unit tests separately from heavier GPU benchmarks:

```bash
scripts/install_faiss.sh
pytest -m "not faiss_gpu"        # unit tests
pytest -m faiss_gpu --strict-markers  # optional heavy job
```

Benchmark results can be recorded with `scripts/bench_all.py --record` to
`benchmarks/<commit>.json` and compared against the previous commit
automatically.
Generate a history table with `scripts/bench_all.py --trend` to create
`benchmarks/bench_trend.md` summarizing throughput across commits.

Upload the cache dashboard to Grafana using `scripts/upload_dashboard.py`.


### 1. Tool Setup

The application persists all state in Redis and Neo4j so no local data
directories are required. User and dataset records are cached in Redis for
quick access. Ensure both services are running and accessible via the
configuration file. If you plan to use the bundled vLLM helpers
start the server as shown below:

```bash
# Start vLLM server
# Note you will need to grab your HF Authentication from: https://huggingface.co/settings/tokens
vllm serve meta-llama/Llama-3.3-70B-Instruct --port 8000
```

### 2. Usage

Start the REST API server and interact with the endpoints.

All operations run asynchronously. Use `/tasks/ingest`, `/tasks/generate`,
`/tasks/curate` and `/tasks/save` to launch long running processes in the
background. Each request returns a `task_id` which can be polled via
`/tasks/{task_id}`.

Tasks are executed by [Celery](https://docs.celeryq.dev/). By default an in-memory
broker is used, but in production you should set `CELERY_BROKER_URL` and
`CELERY_RESULT_BACKEND` to a Redis or RabbitMQ instance.

Datasets can be managed through `/datasets` (create, list, update, delete and
download). Each dataset records its history in Redis. Retrieve the current
progress via `/datasets/<name>/progress` and past events via
`/datasets/<name>/history`. Generation runs are versioned so you can review each
attempt. List them via `/datasets/<name>/versions`, fetch a single run with
`/datasets/<name>/versions/{n}` and remove unwanted entries with
`DELETE /datasets/<name>/versions/{n}`. Each version stores the parameters and
summary information for that generation. Every request must include an
`X-API-Key` header issued when creating a user via `/users`.

To locate relevant chunks using vector similarity, issue a POST request to
`/vector/search` with the dataset name and query string:

```bash
curl -X POST -H "X-API-Key: <token>" -H "Content-Type: application/json" \
     -d '{"dataset":"demo","query":"graph"}' http://localhost:8000/vector/search
```

Or from the browser:

```js
fetch("/vector/search", {
  method: "POST",
  headers: {"X-API-Key": "<token>", "Content-Type": "application/json"},
  body: JSON.stringify({dataset: "demo", query: "graph"})
}).then(r => r.json())
```
Datasets are persisted in Redis and Neo4j only. Use `DatasetBuilder.to_redis` and `save_neo4j` for persistence.

### Architecture Overview

```mermaid
graph TD
    Server(FastAPI) --> API[REST API]
    API --> Builder
    API --> Celery(Celery Worker)
    Celery --> Builder
    Builder[DatasetBuilder] --> Parsers
    Builder --> Generators
    Builder --> LLMClient
    Builder --> FormatConverter
    FormatConverter --> S3[(S3 Storage)]
    Builder --> Analysis
    Builder --> Perception
    Builder --> Security
    Builder --> Cache[(Redis/LMDB Cache)]
    Builder --> KG[KnowledgeGraph]
    KG --> Redis[(Redis DB)]
    KG --> Neo4j[(Neo4j)]

    subgraph Parsers
        PDFParser
        HTMLParser
        YouTubeParser
        DOCXParser
        PPTParser
        TXTParser
        CodeParser
        ImageParser
        AudioParser
        WhisperParser[WhisperAudioParser]
    end

    subgraph Generators
        QAGenerator
        COTGenerator
        ConversationGenerator
        KGGenerator
        VQAGenerator
        PrefGenerator
        ToolGenerator
        MultiToolGenerator
    end

    Config[Configuration] --> API
    Config --> Builder

    Utils --> FormatConverter
    Utils --> Security

    %% Batch processing is handled internally, no separate module

    EnvVars[Environment Variables] -.-> Builder
    ConfigWatcher[[Config Watcher]] -.-> Builder
    PolicyMonitor[[Policy Monitor]] -.-> Builder
```

## Configuration

The toolkit uses a YAML configuration file (default: `configs/config.yaml`).
Database connection settings can be provided either through the
`DATABASE_URL` environment variable or a `database.url` entry in the YAML
file. By default a local SQLite file `datacreek.db` is used, but you can
point this to any SQLAlchemy compatible database.

### Database initialization

Run `python -m datacreek.cli init-db` to create the tables before starting the
server if they do not already exist. The API container executes this step on
startup so the database is ready when the services come online.
The command line interface is reserved for such maintenance operations (database
setup, test runs) and is not intended for dataset generation.

### Development build

Use the source tree directly when hacking on the project. Install the
dependencies from `requirements.txt`:

```bash
git clone https://github.com/meta-llama/datacreek.git
cd datacreek
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The front-end lives in `frontend` and can be started separately for a faster
iteration loop:

```bash
cd frontend
npm install
npm run dev
```

This will watch for file changes and serve the interface on
`http://localhost:5173` while the API runs on port 8000.

### Starting the stack with Docker

Use the provided `docker-compose.yml` to launch the API, Celery worker,
Redis, Neo4j and the front-end:

The stack consists of six main services:

- **api** – FastAPI server exposing ingestion, generation, curation and export
  endpoints.
- **worker** – Celery background worker executing long-running tasks.
- **redis** – message broker and task result backend.
- **neo4j** – knowledge graph database.
- **backend** – Flask web interface.
- **frontend** – React web application built with Vite.

```bash
./scripts/start_services.sh
```

If no `.env` file exists the script will copy `.env.example` so you only
need to adjust values for production. The default configuration stores
the SQLite database and generated datasets in `./data` which is mounted
inside the containers.

The API will be available on `http://localhost:8000` and the Flask backend on
`http://localhost:5000` while the front-end is served on `http://localhost:3000`.
Redis listens on `6379` and Neo4j exposes `7474` and `7687`.

To rebuild the images after modifying the code simply run:

```bash
docker compose build
```

### Deployment

Use the `scripts/deploy.sh` helper to update a remote host. The CI pipeline
builds container images for the API, worker and front-end and pushes them to
GitHub Container Registry. On deployment the remote host pulls the latest
images defined in `.env` and restarts the stack. Export `DEPLOY_HOST`,
`DEPLOY_USER`, `DEPLOY_KEY` and `DEPLOY_PATH` before executing the script.
Docker Compose automatically loads variables from an optional `.env` file
located next to `docker-compose.yml`:

```bash
export DEPLOY_HOST=example.com
export DEPLOY_USER=ubuntu
export DEPLOY_PATH=/opt/datacreek
export DEPLOY_KEY=~/.ssh/id_rsa
scripts/deploy.sh
```

Environment variables can be configured via a `.env` file. See
`.env.example` for defaults.

At a minimum, set `NEO4J_URI`, `NEO4J_USER` and `NEO4J_PASSWORD` so the
API can reach the Neo4j instance. `DATABASE_URL` defaults to storing the
SQLite file inside the mounted `./data` directory. `IMAGE_NAME` and
`FRONTEND_IMAGE_NAME` define the container images pulled during deployment.

Before running the deployment script ensure your CI pipeline built and pushed
the container images to the registry referenced by `IMAGE_NAME` and
`FRONTEND_IMAGE_NAME`. The remote host only pulls and restarts the services;
all configuration is controlled by the `.env` file copied alongside
`docker-compose.yml`.

You can override any value by providing a custom YAML file to the server.

```yaml
# Example configuration using vLLM
llm:
  provider: "vllm"

vllm:
  api_base: "http://localhost:8000/v1"
  model: "meta-llama/Llama-3.3-70B-Instruct"

generation:
  temperature: 0.7
  chunk_size: 4000
  chunk_method: sliding  # basic|sliding|semantic|contextual|summary
  retrieval_top_k: 3
  num_pairs: 25

curate:
  threshold: 7.0
  batch_size: 8
```

or using an API endpoint:

```yaml
# Example configuration using the llama API
llm:
  provider: "api-endpoint"

api-endpoint:
  api_base: "https://api.llama.com/v1"
  api_key: "llama-api-key"
  model: "Llama-4-Maverick-17B-128E-Instruct-FP8"
```

### Customizing Configuration

Create a custom configuration file and pass it via the `X-Config-Path` header:

The `generation` section now exposes advanced chunking and retrieval options:

```
  generation:
    chunk_method: semantic  # or "sliding" for fixed windows, "contextual" or "summary" for prefixed chunks
  similarity_drop: 0.25   # threshold when using semantic splitting
    retrieval_top_k: 5      # number of chunks fetched using embeddings
```

Most options can also be overridden with environment variables. For example set
`GEN_TEMPERATURE=0.5` to change the default temperature.

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `SDK_VERBOSE` | Enable verbose output for all operations | `false` | `export SDK_VERBOSE=true` |
| `SDK_BATCH_SIZE` | Override batch size for curate command | Config setting | `export SDK_BATCH_SIZE=1` |
| `LLM_PROVIDER` | Choose underlying LLM provider | – | `export LLM_PROVIDER=vllm` |
| `LLM_API_BASE` | Base URL for the LLM API | – | `export LLM_API_BASE=http://localhost:8000/v1` |
| `LLM_MODEL` | Model name for LLM calls | – | `export LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct` |
| `LLM_MAX_RETRIES` | Max retries for LLM requests | `3` | `export LLM_MAX_RETRIES=5` |
| `LLM_RETRY_DELAY` | Delay between retries | `1.0` | `export LLM_RETRY_DELAY=2` |
| `API_ENDPOINT_KEY` | API key for external provider | – | `export API_ENDPOINT_KEY=sk-abc` |
| `DATACREEK_CONFIG` | Path to YAML configuration file (hot-reloaded) | – | `export DATACREEK_CONFIG=config.yaml` |
| `DATACREEK_PIPELINES_CONFIG` | Path to pipeline definitions | – | `export DATACREEK_PIPELINES_CONFIG=pipelines.yaml` |
| `REDIS_HOST` | Redis hostname | `localhost` | `export REDIS_HOST=redis` |
| `REDIS_PORT` | Redis port | `6379` | `export REDIS_PORT=6379` |
| `S3_BUCKET` | Upload dataset exports to this bucket | – | `export S3_BUCKET=my-bucket` |
| `S3_PREFIX` | Optional prefix for uploaded objects | – | `export S3_PREFIX=demo/` |
| `S3_ENDPOINT_URL` | Custom S3-compatible endpoint | – | `export S3_ENDPOINT_URL=https://s3.example.com` |
| `AWS_ACCESS_KEY_ID` | AWS credential for S3 uploads | – | `export AWS_ACCESS_KEY_ID=abc` |
| `AWS_SECRET_ACCESS_KEY` | AWS secret for S3 uploads | – | `export AWS_SECRET_ACCESS_KEY=xyz` |
| `USE_REDIS_GRAPH` | Enable RedisGraph integration | `false` | `export USE_REDIS_GRAPH=1` |
| `DATACREEK_REQUIRE_PERSISTENCE` | Require Redis and Neo4j backends | `1` | `export DATACREEK_REQUIRE_PERSISTENCE=0` |
| `DATACREEK_UPLOAD_DIR` | Restrict ingestion to this directory | – | `export DATACREEK_UPLOAD_DIR=/tmp/uploads` |
| `DATABASE_URL` | SQLAlchemy database URL | `sqlite:///datacreek.db` | `export DATABASE_URL=postgresql://user:pass@host/db` |
| `NEO4J_URI` | URI of the Neo4j database | `bolt://localhost:7687` | `export NEO4J_URI=bolt://neo4j:7687` |
| `NEO4J_USER` | Username for Neo4j | `neo4j` | `export NEO4J_USER=neo4j` |
| `NEO4J_PASSWORD` | Password for Neo4j | `neo4j` | `export NEO4J_PASSWORD=pass` |
| `NEO4J_INIT_INDEXES` | Create indexes on startup | `1` | `export NEO4J_INIT_INDEXES=0` |
| `CELERY_BROKER_URL` | Celery message broker | `redis://localhost/0` | `export CELERY_BROKER_URL=redis://redis:6379/0` |
| `CELERY_RESULT_BACKEND` | Celery result backend | `redis://localhost/0` | `export CELERY_RESULT_BACKEND=redis://redis:6379/0` |
| `PORT` | Flask backend port | `5000` | `export PORT=8000` |
| `SECRET_KEY` | Flask secret key | – | `export SECRET_KEY=change-me` |
| `HOST` | Bind address for the FastAPI server | `0.0.0.0` | `export HOST=127.0.0.1` |
| `DEBUG` | Enable FastAPI debug mode | `false` | `export DEBUG=true` |
| `API_KEY` | Default API key for the web interface | – | `export API_KEY=mykey` |
| `IMAGE_NAME` | Container image for the API | – | `export IMAGE_NAME=ghcr.io/org/app` |
| `FRONTEND_IMAGE_NAME` | Container image for the front-end | – | `export FRONTEND_IMAGE_NAME=ghcr.io/org/front` |
| `DEPLOY_HOST` | Remote host used by `deploy.sh` | – | `export DEPLOY_HOST=example.com` |
| `DEPLOY_USER` | SSH user for deployment | – | `export DEPLOY_USER=ubuntu` |
| `DEPLOY_PATH` | Target directory on the remote host | – | `export DEPLOY_PATH=/opt/datacreek` |
| `DEPLOY_KEY` | SSH key used for the connection | – | `export DEPLOY_KEY=~/.ssh/id_rsa` |
| `LOCAL_BUILD` | Build images locally instead of pulling | `false` | `export LOCAL_BUILD=true` |
| `DATASET_MAX_VERSIONS` | Maximum versions kept per dataset | – | `export DATASET_MAX_VERSIONS=5` |
| `SDK_DEBUG` | Log full model responses | – | `export SDK_DEBUG=1` |
### Hot-reload config

Set the environment variable `DATACREEK_CONFIG` to point to your YAML
configuration file. Datacreek watches this file and reloads the `cleanup.*`
section automatically every five minutes, so threshold updates are applied
without restarting the application. A log message `CFG-HOT watcher started`
is emitted on startup. Before each cleanup run the pipeline calls
`verify_thresholds()` to ensure the live values match the YAML file. A mismatch
raises a `RuntimeError`.

### Model Profiles

Define multiple model profiles in your configuration to easily switch between
providers or models:

```yaml
models:
  local-llama:
    provider: vllm
    api_base: "http://localhost:8000/v1"
    model: "meta-llama/Llama-3.3-70B-Instruct"
  llama-api:
    provider: api-endpoint
    api_base: "https://api.llama.com/v1"
    model: "Llama-4-Maverick-17B-128E-Instruct-FP8"
```

Select a profile by passing `profile` when calling the API or constructing an
``LLMClient``.

You can override prompt templates per request by sending a `prompts` object to
`/tasks/generate`:

```bash
curl -X POST localhost:8000/tasks/generate \
     -H "Content-Type: application/json" \
     -H "X-API-Key: <key>" \
     -d '{"src_id": 1, "prompts": {"qa_generation": "Ask three short questions"}}'
```

```bash
curl -X POST localhost:8000/tasks/ingest \
     -H "X-Config-Path: custom_config.yaml" \
     -H "X-API-Key: <key>" \
     -d "path=docs/paper.pdf"
```

## Examples

### Processing a PDF Document

```bash
# Ingest PDF with entity extraction
curl -X POST localhost:8000/tasks/ingest \
     -H "Content-Type: application/json" \
     -H "X-API-Key: <key>" \
     -d '{"path": "research_paper.pdf", "extract_entities": true}'  # unstructured parsing enabled by default

# Generate QA pairs (assuming source ID 1)
curl -X POST localhost:8000/tasks/generate -d "src_id=1&num_pairs=30" -H "X-API-Key: <key>"

# Override the QA generation prompt for this request
curl -X POST localhost:8000/tasks/generate \
     -H "Content-Type: application/json" \
     -H "X-API-Key: <key>" \
     -d '{"src_id": 1, "prompts": {"qa_generation": "Ask three short questions"}}'

# Curate data (dataset ID 1)
curl -X POST localhost:8000/tasks/curate -d "ds_id=1&threshold=8.5" -H "X-API-Key: <key>"

# Save in OpenAI fine-tuning format
curl -X POST localhost:8000/tasks/save -d "ds_id=1&fmt=jsonl" -H "X-API-Key: <key>"
```

### Processing a YouTube Video

```bash
# Extract transcript and generate QA pairs
curl -X POST localhost:8000/tasks/ingest -d "path=https://www.youtube.com/watch?v=dQw4w9WgXcQ" -H "X-API-Key: <key>"
curl -X POST localhost:8000/tasks/generate -d "src_id=1" -H "X-API-Key: <key>"
```

### Processing Multiple Files

```bash
# Bash script to process multiple files
for file in /path/to/pdfs/*.pdf; do
  filename=$(basename "$file" .pdf)

  curl -X POST localhost:8000/tasks/ingest -d "path=$file" -H "X-API-Key: <key>"
  curl -X POST localhost:8000/tasks/generate -d "src_id=1&num_pairs=20" -H "X-API-Key: <key>"
  curl -X POST localhost:8000/tasks/curate -d "ds_id=1&threshold=7.5" -H "X-API-Key: <key>"
  curl -X POST localhost:8000/tasks/save -d "ds_id=1&fmt=chatml" -H "X-API-Key: <key>"
done
```

## Advanced Usage

### Custom Prompt Templates

Edit the `prompts` section in your configuration file to customize generation behavior:

```yaml
prompts:
  qa_generation: |
    You are creating question-answer pairs for fine-tuning a legal assistant.
    Focus on technical legal concepts, precedents, and statutory interpretation.
    
    Below is a chunk of text about: {summary}...
    
    Create {num_pairs} high-quality question-answer pairs based ONLY on this text.
    
    Return ONLY valid JSON formatted as:
    [
      {
        "question": "Detailed legal question?",
        "answer": "Precise legal answer."
      },
      ...
    ]
    
    Text:
    ---
    {text}
    ---
```

Each dataset you create owns its own knowledge graph. During ingestion the
selected documents are inserted into this graph and linked to their original
source.  Generation steps query this cleaned graph instead of the raw files.
The graph exposes simple search helpers so you can explore the content:

```python
import networkx as nx

from datacreek import DatasetBuilder, DatasetType, KnowledgeGraph

ds = DatasetBuilder(DatasetType.QA, name="example")
ds.add_document("doc1", source="paper.pdf")
ds.add_chunk("doc1", "c1", "hello world")
ds.add_image("doc1", "img0", "pages/img0.png", page=1)
print(ds.search("hello"))  # ["c1"]
print(ds.search_documents("paper"))  # ["doc1"]
print(ds.get_chunks_for_document("doc1"))  # ["c1"]
print(ds.get_images_for_document("doc1"))  # ["img0"]
print(ds.get_document_for_chunk("c1"))  # "doc1"
ds.graph.index.build()
print(ds.graph.search_embeddings("hello", k=1))  # ["c1"]
print(ds.graph.search_hybrid("hello"))  # ["c1"]
print(ds.search_hybrid("paper", node_type="document"))  # ["doc1"]
print(ds.similar_by_hybrid("c1"))  # [("c2", 0.9), ...]
print(ds.ann_hybrid_search(n2v_vec, gw_vec, hyp_vec))  # [("c5", 0.8), ...]
print(ds.apply_k_out_privacy(["c1", "c2", "c3"]))
print(ds.multiview_contrastive_loss())  # InfoNCE loss
ds.compute_meta_embeddings()
ds.compute_product_manifold_embeddings()
ds.train_product_manifold_embeddings([("c1", "c2")])
ds.compute_aligned_cca_embeddings()
ds.compute_hyper_sagnn_head_drop_embeddings()
print(ds.hybrid_score("c1", "c2"))
print(ds.governance_metrics())
g2 = nx.cycle_graph(3)
print(ds.persistence_wasserstein_distance(g2))
print(ds.tpl_correct_graph(g2))
print(ds.sheaf_consistency_score())
print(ds.search_with_links("hello", hops=1))  # ["c1", "c2", ...]
print(ds.search_with_links_data("hello", hops=1)[0])  # includes depth and path
ds.link_similar_chunks()         # connect semantically close chunks
ds.update_embeddings()           # materialize embeddings on graph nodes
ds.extract_facts()               # populate fact nodes using an LLM or regex
fact_id = ds.get_facts_for_chunk("c1")[0]
print(ds.get_documents_for_fact(fact_id))  # ["doc1"]
print(ds.find_conflicting_facts())  # check for conflicting information

# Hyperbolic reasoning utilities
print(ds.hyperbolic_neighbors("c1"))
print(ds.hyperbolic_reasoning("c1", "c5"))

# After ingestion you can further enrich the graph:
ds.consolidate_schema()        # normalize labels
ds.detect_communities()        # cluster chunks
ds.summarize_communities()     # generate simple summaries
ds.detect_entity_groups()      # cluster entities
ds.summarize_entity_groups()   # summarize entity groups
ds.score_trust()               # compute naive trust scores
# provenance is stored on edges so you can track sources

Files can also be ingested directly via the REST API:

```bash
curl -X POST localhost:8000/api/datasets/example/ingest \
     -H "Content-Type: application/json" \
     -H "X-API-Key: <key>" \
     -d '{"path": "paper.pdf", "high_res": true, "ocr": true,
          "extract_entities": true, "extract_facts": true}'  # unstructured is default
```

You can then enrich and query the graph via the API:

```bash
curl -X POST localhost:8000/api/datasets/example/similarity -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/search_hybrid?q=hello -H "X-API-Key: <key>"
curl -X GET  "localhost:8000/api/datasets/example/search_hybrid?q=paper&type=document" -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/search_links?q=hello\&hops=1 -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/similar_chunks?cid=c1 -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/similar_chunks_data?cid=c1 -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/chunk_neighbors -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/chunk_neighbors_data -H "X-API-Key: <key>"
The chunk_neighbors_data endpoint returns similarity scores, neighbor text, and the owning document for each chunk.
curl -X POST localhost:8000/api/datasets/example/section_similarity -H "X-API-Key: <key>"
curl -X POST localhost:8000/api/datasets/example/document_similarity -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/chunk_context?cid=c1 -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/similar_sections?sid=s1 -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/similar_documents?did=doc1 -H "X-API-Key: <key>"
curl -X POST localhost:8000/api/datasets/example/extract_facts -H "X-API-Key: <key>"
curl -X POST localhost:8000/api/datasets/example/extract_entities -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/conflicts -H "X-API-Key: <key>"
curl -X POST localhost:8000/api/datasets/example/mark_conflicts -H "X-API-Key: <key>"
curl -X POST localhost:8000/api/datasets/example/prune -H "X-API-Key: <key>" \
     -d '{"sources": ["bad_source"]}'
curl -X POST localhost:8000/api/datasets/example/deduplicate -H "X-API-Key: <key>"
curl -X POST localhost:8000/api/datasets/example/clean_chunks -H "X-API-Key: <key>"
curl -X POST localhost:8000/api/datasets/example/normalize_dates -H "X-API-Key: <key>"
curl -X POST localhost:8000/api/datasets/example/co_mentions -H "X-API-Key: <key>"
curl -X POST localhost:8000/api/datasets/example/doc_co_mentions -H "X-API-Key: <key>"
curl -X POST localhost:8000/api/datasets/example/section_co_mentions -H "X-API-Key: <key>"
curl -X POST localhost:8000/api/datasets/example/graph_embeddings -H "X-API-Key: <key>" \
     -d '{"dimensions": 32, "walk_length": 5, "num_walks": 20, "seed": 42}'
curl -X GET  localhost:8000/api/datasets/example/chunk_document?cid=c1 -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/chunk_page?cid=c1 -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/section_page?sid=s1 -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/fact_documents?fid=f1 -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/fact_pages?fid=f1 -H "X-API-Key: <key>"
curl -X GET  localhost:8000/api/datasets/example/entity_pages?eid=A -H "X-API-Key: <key>"
```

Provide either a local path or a URL directly. Ingestion no longer relies on
configured directories and simply reads the resource you specify.

# Clone a dataset to experiment with different cleaning steps
ds_copy = ds.clone(name="copy")


## Dataset Generation Pipelines

After ingestion, parsed content is placed into a knowledge graph. Generation
pipelines operate on this graph and are specialized for different training goals.

| Dataset type | Compatible trainings |
|--------------|---------------------|
| `qa`         | SFT, DPO, ORPO, DPO+SFT, PPO, RRHF, RLAIF, GRPO |
| `cot`        | SFT, DPO, ORPO, DPO+SFT, RRHF |
| `vqa`        | SFT |
| `text`       | CPT |
| `kg`         | SFT, DPO, ORPO, DPO+SFT, PPO, RRHF, RLAIF, GRPO |
| `pref_pair`  | PPO, DPO, ORPO, DPO+SFT, RLAIF |
| `pref_list`  | GRPO, RRHF |
| `tool`       | SFT, DPO, ORPO, DPO+SFT, PPO, RRHF, RLAIF, GRPO |
| `conversation` | SFT, DPO, ORPO, DPO+SFT, PPO, RRHF, RLAIF, GRPO |
| `multi_tool` | SFT, DPO, ORPO, DPO+SFT, PPO, RRHF, RLAIF, GRPO |

### Training-Specific Pipelines

Depending on the training objective, different dataset types are available.

| Training goal | Supported dataset types |
|---------------|-----------------------|
| SFT | qa, cot, vqa, kg, tool, conversation, multi_tool |
| DPO | qa, cot, kg, pref_pair, tool, conversation, multi_tool |
| ORPO | qa, cot, kg, pref_pair, tool, conversation, multi_tool |
| DPO+SFT | qa, cot, kg, pref_pair, tool, conversation, multi_tool |
| PPO | qa, kg, pref_pair, tool, conversation, multi_tool |
| RRHF | qa, cot, kg, pref_list, tool, conversation, multi_tool |
| RLAIF | qa, kg, pref_pair, tool, conversation, multi_tool |
| GRPO | qa, kg, pref_list, tool, conversation, multi_tool |
| CPT | text |

```python
from datacreek import TrainingGoal, get_dataset_types_for_training

print(get_dataset_types_for_training(TrainingGoal.DPO))
```

You can query pipelines programmatically:

```python
from datacreek import TrainingGoal, get_pipelines_for_training

print(get_pipelines_for_training(TrainingGoal.SFT))
```

Use `run_generation_pipeline` to execute the generation steps directly on a
knowledge graph:

```python
from datacreek import (
    DatasetType,
    KnowledgeGraph,
    run_generation_pipeline,
    run_generation_pipeline_async,
)

kg = KnowledgeGraph()
kg.add_document("doc", source="text", text="Hello world")

# Synchronous usage
qa_data = run_generation_pipeline(DatasetType.QA, kg)

# Asynchronous usage
# qa_data = await run_generation_pipeline_async(DatasetType.QA, kg)
Both functions raise `PipelineExecutionError` when a step fails.
```

## Post-Ingestion Operations

After parsing documents into the knowledge graph you can refine the data quality
with a few helper methods:

- `deduplicate_chunks()` removes chunks with identical text
- `resolve_entities(aliases={...})` merges entity nodes referring to the same concept and accepts a case-insensitive alias mapping
- `prune_sources(['src'])` deletes nodes originating from unwanted sources
- `link_chunks_by_entity()` connects chunks that mention the same entity
- `link_sections_by_entity()` connects sections that mention the same entity
- `link_documents_by_entity()` connects documents that mention the same entity
- `link_authors_organizations()` links document authors to their organizations
- `clean_chunk_texts()` removes HTML tags and excess whitespace from chunks
- `normalize_date_fields()` standardizes date attributes to ISO format
- `compute_graph_embeddings(dimensions=64, walk_length=10, num_walks=50, seed=0, workers=2)` generates Node2Vec embeddings for all nodes. Adjust these parameters to tune vector size and random walks
- `predict_links(use_graph_embeddings=True)` infers relations using graph embeddings
- `mark_conflicting_facts()` flags edges when multiple objects disagree
- `validate_coherence()` marks logically impossible relations like a parent born after a child
- `apply_perception()` updates a node and automatically runs Neo4j `gds.nodeSimilarity` to flag near duplicates
- `apply_perception_all_nodes()` transforms every node and performs the same similarity check
- `node_similarity(id, threshold=0.95)` returns nodes similar to a given ID using Neo4j GDS
- `enrich_entity_wikidata(id)` fetches label and description from Wikidata
- `enrich_entity_dbpedia(id)` fetches additional info from DBpedia

Example:

```python
builder.apply_perception(
    "chunk_1",
    "Shortened text.",
    perception_id="summary",
    strength=0.7,
    threshold=0.9,
)

similar = builder.node_similarity("chunk_1", threshold=0.95)
print(similar)
```

Each similarity query is stored as a `node_similarity_check` event when matches are found, and every call to `node_similarity()` emits a `node_similarity_query` event for traceability.

These utilities are exposed through both the REST API and the web interface.

## Troubleshooting FAQs:

### vLLM Server Issues

- Ensure vLLM is installed: `pip install vllm`
- Start server with: `vllm serve <model_name> --port 8000`
- Check connection: `curl http://localhost:8000/docs`

### Memory Issues

If you encounter CUDA out of memory errors:
- Use a smaller model
- Reduce batch size in config
- Start vLLM with `--gpu-memory-utilization 0.85`

### JSON Parsing Issues

If you encounter issues during the curation step:
- Enable verbose logging
- Set smaller batch sizes in your config.yaml
- Ensure the LLM model supports proper JSON output
- Install json5 for enhanced JSON parsing: `pip install json5`

### Parser Errors

- Ensure required dependencies are installed for specific parsers:
  - YouTube: `pip install pytubefix youtube-transcript-api`
  - Office/PDF/HTML: `pip install "unstructured[all-docs]"`

## Web Interface

A lightweight React application built with Vite lives in the `frontend`
directory. It uses Tailwind CSS v4 for styling and communicates with the Flask
API for authentication and dataset operations.

```bash
cd frontend
npm install
npm run dev  # start the Vite dev server
```

Point your browser to `http://localhost:5173` while the Flask API runs on
`http://localhost:8000`.

## SaaS Pipeline Logic

The service orchestrates several stages to turn raw content into fine‑tuning
records.

1. **Template selection** – `get_template()` loads a prompt template definition
   with schema constraints and maximum length. Responses are validated with
   `validate_output()` before being accepted.
2. **Hybrid search** – `search_with_links()` combines lexical lookup with
   embedding neighbors. The optional `fractal_level` parameter restricts
   traversal depth for coarse or fine context.
3. **Graph‑to‑text encoding** – `neighborhood_to_sentence()` transforms graph
   paths into sentences, recursively summarizing neighbors for richer prompts.
4. **Self‑Instruct generation** – `generate_with_self_instruct()` repeatedly
   queries the LLM until the template validation succeeds. Tool call examples can
   be inserted via `auto_tool_calls()`.
5. **Fact verification** – `verify_statements()` checks generated triples against
   the knowledge graph using shortest paths to assign a confidence score.
6. **Diversification** – `select_diverse_nodes()` picks nodes that maximize the
   `diversification_score()` ensuring coverage of new graph regions.
7. **Fractal metrics** – `embedding_box_counting_dimension()` and
   `dimension_distortion()` measure how well the hyperbolic embeddings
   capture the topology. Use them with a list of radii to monitor
   geometry drift.
8. **Prompt export & coverage** – `export_prompts()` attaches a MD5
   `signature_hash` of the topological signature and a `prompt_hash` so
   duplicates can be detected. Edge traversal is recorded so
   `coverage_stats()` can report how much of the graph has been explored.
   Supplying an `encrypt_key` masks PII in the exported records. User
   feedback is stored via `record_feedback()` and merge/split suggestions can
   be generated with `propose_merge_split()` for continuous improvements.

9. **LLM service connectors** – `LLMService` centralizes OpenAI/vLLM access and
   provides synchronous and asynchronous batch completions.

## Compression

*Si la perplexité après pruning dépasse 1 % de la référence, le pruner restaure automatiquement le checkpoint précédent.*

### Rollback policy

| Scenario | Action |
| -------- | ------ |
| FractalNet pruning ratio > 1.01 | Restore previous checkpoint and increment `prune_reverts_total` |
| DP budget exceeded | Abort operation and log `DP_ROLLBACK=true` |

## License

Read more about the [License](./LICENSE)

## Contributing

Contributions are welcome! [Read our contributing guide](./CONTRIBUTING.md)

