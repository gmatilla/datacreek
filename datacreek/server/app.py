# pragma: no cover
"""Flask application for the Datacreek web interface.

This module starts a full Flask server with many optional dependencies
and is excluded from coverage due to its heavy runtime requirements.
"""

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import (
    LoginManager,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_wtf import FlaskForm
from werkzeug.security import check_password_hash, generate_password_hash
from wtforms import (
    FileField,
    IntegerField,
    PasswordField,
    SelectField,
    StringField,
    SubmitField,
)
from wtforms.validators import DataRequired

logger = logging.getLogger(__name__)

from datacreek.backends import get_neo4j_driver as backend_get_neo4j_driver
from datacreek.backends import get_redis_client as backend_get_redis_client
from datacreek.core.create import process_file
from datacreek.core.curate import curate_qa_pairs
from datacreek.core.dataset import DatasetBuilder
from datacreek.core.ingest import ingest_into_dataset
from datacreek.core.ingest import process_file as ingest_process_file
from datacreek.core.knowledge_graph import KnowledgeGraph
from datacreek.db import SessionLocal, User, init_db
from datacreek.models.export_format import ExportFormat
from datacreek.models.llm_client import LLMClient
from datacreek.models.stage import DatasetStage
from datacreek.pipelines import DatasetType
from datacreek.services import generate_api_key, hash_key
from datacreek.tasks import (
    dataset_cleanup_task,
    dataset_delete_task,
    dataset_export_task,
    dataset_extract_entities_task,
    dataset_extract_facts_task,
    dataset_generate_task,
    dataset_ingest_task,
    dataset_load_neo4j_task,
    dataset_operation_task,
    dataset_save_neo4j_task,
    graph_delete_task,
    graph_load_neo4j_task,
    graph_save_neo4j_task,
)
from datacreek.utils import backpressure
from datacreek.utils.config import get_llm_provider, load_config
from datacreek.utils.neo4j_breaker import neo4j_breaker

STATIC_DIR = Path(__file__).parents[2] / "frontend" / "dist"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/")
app.config["SECRET_KEY"] = os.urandom(24)

login_manager = LoginManager(app)
login_manager.login_view = None


@login_manager.unauthorized_handler
def unauthorized():
    return jsonify({"error": "login required"}), 401


# Load SDK config
config = load_config()
init_db()


def get_redis_client():
    """Return a Redis client using backend helpers."""
    try:
        client = backend_get_redis_client()
        client.ping()
        return client
    except Exception:
        logger.exception("Failed to connect to Redis")
        return None


REDIS = get_redis_client()


def load_datasets_from_redis() -> None:
    """Populate ``DATASETS`` from Redis if available."""
    if not REDIS:
        return
    for name in REDIS.smembers("datasets"):
        try:
            driver = get_neo4j_driver()
            ds = DatasetBuilder.from_redis(REDIS, f"dataset:{name}", driver)
            ds.redis_client = REDIS
            ds.neo4j_driver = driver
            DATASETS[name] = ds
        except KeyError:
            continue


# defer dataset loading until first access to avoid holding
# all data in memory when running as a SaaS


def load_graphs_from_redis() -> None:
    """Populate ``GRAPHS`` from Redis if available."""
    if not REDIS:
        return
    for name in REDIS.smembers("graphs"):
        try:
            driver = get_neo4j_driver()
            ds = DatasetBuilder.from_redis(REDIS, f"graph:{name}", driver)
            ds.redis_client = REDIS
            ds.neo4j_driver = driver
            GRAPHS[name] = ds
        except KeyError:
            continue


# defer graph loading until accessed to keep startup fast


@login_manager.user_loader
def load_user(user_id: str) -> User | None:
    with SessionLocal() as db:
        return db.get(User, int(user_id))


def get_neo4j_driver():
    """Return a Neo4j driver using backend helpers."""
    return backend_get_neo4j_driver()


# In-memory store of datasets being built
DATASETS: Dict[str, DatasetBuilder] = {}
# Store knowledge graphs independently from datasets
GRAPHS: Dict[str, DatasetBuilder] = {}

# Datasets and graphs are loaded lazily from Redis on first access


def get_dataset(name: str) -> DatasetBuilder | None:
    """Return dataset ``name`` loading from Redis if necessary."""
    ds = DATASETS.get(name)
    if ds:
        if current_user.is_authenticated and ds.owner_id not in {None, current_user.id}:
            return None
        return ds
    if REDIS:
        if current_user.is_authenticated:
            key = f"user:{current_user.id}:datasets"
            if REDIS.exists(key) and not REDIS.sismember(key, name):
                return None
        try:
            driver = get_neo4j_driver()
            ds = DatasetBuilder.from_redis(REDIS, f"dataset:{name}", driver)
            ds.redis_client = REDIS
            ds.neo4j_driver = driver
            if current_user.is_authenticated and ds.owner_id not in {
                None,
                current_user.id,
            }:
                return None
            DATASETS[name] = ds
            return ds
        except KeyError:
            return None
    return None


def save_dataset(ds: DatasetBuilder) -> None:
    """Persist ``ds`` to Redis if available."""
    if REDIS:
        ds.redis_client = REDIS
        ds.to_redis(REDIS, f"dataset:{ds.name}")
        REDIS.sadd("datasets", ds.name)
        if ds.owner_id is not None:
            REDIS.sadd(f"user:{ds.owner_id}:datasets", ds.name)


def delete_dataset_persist(name: str) -> None:
    """Remove dataset ``name`` from Redis."""
    if REDIS:
        raw = REDIS.get(f"dataset:{name}")
        owner = None
        if raw:
            try:
                owner = json.loads(raw).get("owner_id")
            except Exception:
                pass
        REDIS.delete(f"dataset:{name}")
        REDIS.srem("datasets", name)
        if owner is not None:
            REDIS.srem(f"user:{owner}:datasets", name)


def get_graph(name: str) -> DatasetBuilder | None:
    """Return knowledge graph ``name`` loading from Redis if necessary."""
    ds = GRAPHS.get(name)
    if ds:
        if current_user.is_authenticated and ds.owner_id not in {None, current_user.id}:
            return None
        return ds
    if REDIS:
        if current_user.is_authenticated:
            key = f"user:{current_user.id}:graphs"
            if REDIS.exists(key) and not REDIS.sismember(key, name):
                return None
        try:
            driver = get_neo4j_driver()
            ds = DatasetBuilder.from_redis(REDIS, f"graph:{name}", driver)
            ds.redis_client = REDIS
            ds.neo4j_driver = driver
            if current_user.is_authenticated and ds.owner_id not in {
                None,
                current_user.id,
            }:
                return None
            GRAPHS[name] = ds
            return ds
        except KeyError:
            return None
    return None


def save_graph(ds: DatasetBuilder) -> None:
    """Persist ``ds`` under the ``graph:`` prefix in Redis."""
    if REDIS:
        ds.redis_client = REDIS
        ds.to_redis(REDIS, f"graph:{ds.name}")
        REDIS.sadd("graphs", ds.name)
        if ds.owner_id is not None:
            REDIS.sadd(f"user:{ds.owner_id}:graphs", ds.name)


def delete_graph_persist(name: str) -> None:
    """Remove graph ``name`` from Redis."""
    if REDIS:
        raw = REDIS.get(f"graph:{name}")
        owner = None
        if raw:
            try:
                owner = json.loads(raw).get("owner_id")
            except Exception:
                pass
        for key in list(REDIS.scan_iter(match=f"graph:{name}*")):
            REDIS.delete(key)
        REDIS.srem("graphs", name)
        if owner is not None:
            REDIS.srem(f"user:{owner}:graphs", name)


# Forms
class CreateForm(FlaskForm):
    """Form for creating content from text"""

    input_file = StringField("Input File Path", validators=[DataRequired()])
    content_type = SelectField(
        "Content Type",
        choices=[
            ("qa", "Question-Answer Pairs"),
            ("summary", "Summary"),
            ("cot", "Chain of Thought"),
            ("cot-enhance", "CoT Enhancement"),
        ],
        default="qa",
    )
    num_pairs = IntegerField("Number of QA Pairs", default=10)
    provider = SelectField(
        "Provider",
        choices=[("vllm", "vLLM"), ("api-endpoint", "API Endpoint")],
        default="vllm",
    )
    model = StringField("Model Name (optional)")
    api_base = StringField("API Base URL (optional)")
    submit = SubmitField("Generate Content")


class IngestForm(FlaskForm):
    """Form for ingesting documents"""

    input_type = SelectField(
        "Input Type",
        choices=[("file", "Upload File"), ("url", "URL"), ("path", "Local Path")],
        default="file",
    )
    upload_file = FileField("Upload Document")
    input_path = StringField("File Path or URL")
    submit = SubmitField("Parse Document")


class CurateForm(FlaskForm):
    """Form for curating QA pairs"""

    input_file = StringField("Input JSON File Path", validators=[DataRequired()])
    num_pairs = IntegerField("Number of QA Pairs to Keep", default=0)
    provider = SelectField(
        "Provider",
        choices=[("vllm", "vLLM"), ("api-endpoint", "API Endpoint")],
        default="vllm",
    )
    model = StringField("Model Name (optional)")
    api_base = StringField("API Base URL (optional)")
    submit = SubmitField("Curate QA Pairs")


class DatasetForm(FlaskForm):
    """Form for creating a new dataset"""

    name = StringField("Dataset Name", validators=[DataRequired()])
    dataset_type = SelectField(
        "Dataset Type",
        choices=[(dt.value, dt.value) for dt in DatasetType],
        default=DatasetType.QA.value,
    )
    submit = SubmitField("Create Dataset")


# API Routes


@app.post("/api/login")
def api_login():
    data = request.get_json() or {}
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"error": "missing credentials"}), 400
    with SessionLocal() as db:
        user = db.query(User).filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return jsonify({"message": "logged in"})
    return jsonify({"error": "invalid credentials"}), 401


@app.post("/api/logout")
@login_required
def api_logout():
    logout_user()
    return jsonify({"message": "logged out"})


@app.post("/api/register")
def api_register():
    data = request.get_json() or {}
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"error": "missing credentials"}), 400
    with SessionLocal() as db:
        if db.query(User).filter_by(username=username).first():
            return jsonify({"error": "username exists"}), 400
        api_key = generate_api_key()
        user = User(
            username=username,
            api_key=hash_key(api_key),
            password_hash=generate_password_hash(password),
        )
        db.add(user)
        db.commit()
    return jsonify({"message": "account created", "api_key": api_key})


@app.get("/api/session")
def api_session():
    if current_user.is_authenticated:
        return jsonify({"username": current_user.username})
    return jsonify({"username": None})


@app.get("/api/datasets")
@login_required
def api_datasets():
    """Return list of dataset names owned by the current user."""

    names: set[str] = set()
    if REDIS:
        key = f"user:{current_user.id}:datasets"
        if REDIS.exists(key):
            for raw in REDIS.smembers(key):
                names.add(raw.decode() if isinstance(raw, bytes) else raw)
        else:
            for raw in REDIS.smembers("datasets"):
                name = raw.decode() if isinstance(raw, bytes) else raw
                try:
                    driver = get_neo4j_driver()
                    ds = DatasetBuilder.from_redis(REDIS, f"dataset:{name}", driver)
                except KeyError:
                    continue
                if ds.owner_id in {None, current_user.id}:
                    names.add(name)

    return jsonify(sorted(names))


@app.post("/api/datasets")
@login_required
def api_create_dataset():
    data = request.get_json() or {}
    name = data.get("name")
    dtype = data.get("dataset_type")
    graph_name = data.get("graph")
    if not name or not dtype:
        abort(400)
    if REDIS and REDIS.exists(f"dataset:{name}"):
        abort(400, description="Dataset already exists")
    if name in DATASETS:
        abort(400, description="Dataset already exists")
    try:
        ds = DatasetBuilder(DatasetType(dtype), name=name)
        ds.owner_id = current_user.id
        ds.redis_client = REDIS
        ds.neo4j_driver = get_neo4j_driver()
    except ValueError:
        abort(400)
    if graph_name:
        g = get_graph(graph_name)
        if not g:
            abort(404)
        ds.graph = KnowledgeGraph.from_dict(g.graph.to_dict())
        ds.stage = max(ds.stage, DatasetStage.INGESTED)
        ds.history.append(f"Initialized from graph {graph_name}")
    ds.history.append("Dataset created")
    DATASETS[name] = ds
    save_dataset(ds)
    return jsonify({"message": "Dataset created"})


@app.get("/api/datasets/<name>")
@login_required
def api_dataset_detail(name: str):
    """Return dataset details as JSON."""
    ds = get_dataset(name)
    if not ds:
        abort(404)
    nodes = ds.graph.graph
    num_docs = sum(1 for _, d in nodes.nodes(data=True) if d.get("type") == "document")
    num_chunks = sum(1 for _, d in nodes.nodes(data=True) if d.get("type") == "chunk")
    num_facts = sum(1 for _, d in nodes.nodes(data=True) if d.get("type") == "fact")
    size = sum(
        len(nodes.nodes[n].get("text", ""))
        for n, d in nodes.nodes(data=True)
        if d.get("type") == "chunk"
    )
    quality = min(100, num_chunks * 5)
    tips = []
    if num_docs == 0:
        tips.append("Add documents to your dataset")
    if num_chunks == 0:
        tips.append("Ingest text chunks to improve quality")
    info = {
        "id": ds.id,
        "name": ds.name,
        "type": ds.dataset_type.value,
        "created_at": ds.created_at.isoformat(),
        "num_nodes": len(nodes.nodes),
        "num_edges": len(nodes.edges),
        "num_documents": num_docs,
        "num_chunks": num_chunks,
        "num_facts": num_facts,
        "size": size,
        "quality": quality,
        "tips": tips,
        "history": ds.history,
        "versions": ds.versions,
        "stage": ds.stage,
    }
    return jsonify(info)


@app.get("/api/datasets/<name>/content")
@login_required
def api_dataset_content(name: str):
    """Return dataset documents with their chunks."""
    ds = get_dataset(name)
    if not ds:
        abort(404)
    docs = []
    for node, data in ds.graph.graph.nodes(data=True):
        if data.get("type") == "document":
            doc = {
                "id": node,
                "source": data.get("source"),
                "chunks": [],
            }
            for cid in ds.graph.get_chunks_for_document(node):
                cdata = ds.graph.graph.nodes[cid]
                doc["chunks"].append({"id": cid, "text": cdata.get("text", "")})
            docs.append(doc)
    return jsonify(docs)


@app.post("/api/datasets/<name>/ingest")
@login_required
def api_dataset_ingest(name: str):
    """Ingest a document into the dataset asynchronously."""
    ds = get_dataset(name)
    if not ds:
        abort(404)
    data = request.get_json() or {}
    path = data.get("path")
    if not path:
        abort(400)
    if not backpressure.has_capacity():
        abort(429)

    tid = dataset_ingest_task.delay(
        name,
        path,
        current_user.id,
        doc_id=data.get("doc_id"),
        config=data.get("config"),
        high_res=bool(data.get("high_res", False)),
        ocr=bool(data.get("ocr", False)),
        use_unstructured=data.get("use_unstructured"),
        extract_entities=bool(data.get("extract_entities", False)),
        extract_facts=bool(data.get("extract_facts", False)),
    ).id
    return jsonify({"task_id": tid})


@app.post("/api/datasets/<name>/generate")
@login_required
def api_dataset_generate(name: str):
    """Run the generation pipeline asynchronously."""
    ds = get_dataset(name)
    if not ds:
        abort(404)
    data = request.get_json() or {}
    params = data.get("params", {})
    if "start_step" in params and isinstance(params["start_step"], str):
        params["start_step"] = PipelineStep(params["start_step"])
    tid = dataset_generate_task.delay(name, params, current_user.id).id
    return jsonify({"task_id": tid})


@app.delete("/api/datasets/<name>")
@login_required
def api_delete_dataset(name: str):
    """Delete the dataset asynchronously."""

    ds = get_dataset(name)
    if not ds:
        abort(404)
    DATASETS.pop(name, None)
    tid = dataset_delete_task.delay(name, current_user.id).id
    return jsonify({"task_id": tid})


@app.delete("/api/datasets/<name>/chunks/<cid>")
@login_required
def api_delete_chunk(name: str, cid: str):
    ds = get_dataset(name)
    if not ds:
        abort(404)
    ds.remove_chunk(cid)
    save_dataset(ds)
    return jsonify({"message": "deleted"})


@app.post("/api/datasets/<name>/deduplicate")
@login_required
def api_deduplicate(name: str):
    """Remove duplicate chunks from the dataset."""
    if not get_dataset(name):
        abort(404)
    tid = dataset_operation_task.delay(
        name, "deduplicate_chunks", None, current_user.id
    ).id
    return jsonify({"task_id": tid})


@app.post("/api/datasets/<name>/clean_chunks")
@login_required
def api_clean_chunks(name: str):
    """Normalize chunk text by stripping markup and whitespace."""
    if not get_dataset(name):
        abort(404)
    tid = dataset_operation_task.delay(name, "clean_chunks", None, current_user.id).id
    return jsonify({"task_id": tid})


@app.post("/api/datasets/<name>/normalize_dates")
@login_required
def api_normalize_dates(name: str):
    """Normalize date fields on dataset nodes."""
    if not get_dataset(name):
        abort(404)
    tid = dataset_operation_task.delay(
        name, "normalize_dates", None, current_user.id
    ).id
    return jsonify({"task_id": tid})


@app.post("/api/datasets/<name>/prune")
@login_required
def api_prune(name: str):
    """Remove nodes originating from specific sources."""
    if not get_dataset(name):
        abort(404)

    data = request.get_json() or {}
    sources = data.get("sources", [])
    if not isinstance(sources, list) or not sources:
        abort(400, description="sources list required")

    tid = dataset_operation_task.delay(
        name,
        "prune_sources",
        {"sources": sources},
        current_user.id,
    ).id
    return jsonify({"task_id": tid})


@app.post("/api/datasets/<name>/cleanup")
@login_required
def api_cleanup_dataset(name: str):
    """Run cleanup operations asynchronously."""
    if not get_dataset(name):
        abort(404)
    data = request.get_json() or {}
    params: dict[str, Any] = {}
    if "resolve_threshold" in data:
        params["resolve_threshold"] = float(data.get("resolve_threshold", 0.8))
    if "resolve_aliases" in data and isinstance(data["resolve_aliases"], dict):
        params["resolve_aliases"] = data["resolve_aliases"]
    if "dedup_similarity" in data:
        params["dedup_similarity"] = float(data.get("dedup_similarity", 1.0))
    if "normalize_dates" in data:
        params["normalize_dates"] = bool(data.get("normalize_dates", True))
    if "mark_conflicts" in data:
        params["mark_conflicts"] = bool(data.get("mark_conflicts", False))
    if "validate" in data:
        params["validate"] = bool(data.get("validate", False))

    tid = dataset_cleanup_task.delay(name, params, current_user.id).id
    return jsonify({"task_id": tid})


@app.post("/api/datasets/<name>/similarity")
@login_required
def api_similarity(name: str):
    """Create similarity links between chunks."""
    if not get_dataset(name):
        abort(404)
    k = int(request.args.get("k", 3))
    tid = dataset_operation_task.delay(
        name, "link_similar_chunks", {"k": k}, current_user.id
    ).id
    return jsonify({"task_id": tid})


@app.post("/api/datasets/<name>/section_similarity")
@login_required
def api_section_similarity(name: str):
    """Create similarity links between section titles."""

    if not get_dataset(name):
        abort(404)
    k = int(request.args.get("k", 3))
    tid = dataset_operation_task.delay(
        name, "link_similar_sections", {"k": k}, current_user.id
    ).id
    return jsonify({"task_id": tid})


@app.post("/api/datasets/<name>/document_similarity")
@login_required
def api_document_similarity(name: str):
    """Create similarity links between documents."""

    if not get_dataset(name):
        abort(404)
    k = int(request.args.get("k", 3))
    tid = dataset_operation_task.delay(
        name, "link_similar_documents", {"k": k}, current_user.id
    ).id
    return jsonify({"task_id": tid})


@app.post("/api/datasets/<name>/co_mentions")
@login_required
def api_co_mentions(name: str):
    """Create links between chunks that mention the same entity."""

    if not get_dataset(name):
        abort(404)
    tid = dataset_operation_task.delay(
        name, "link_chunks_by_entity", None, current_user.id
    ).id
    return jsonify({"task_id": tid})


@app.post("/api/datasets/<name>/doc_co_mentions")
@login_required
def api_doc_co_mentions(name: str):
    """Create links between documents that mention the same entity."""

    if not get_dataset(name):
        abort(404)
    tid = dataset_operation_task.delay(
        name, "link_documents_by_entity", None, current_user.id
    ).id
    return jsonify({"task_id": tid})


@app.post("/api/datasets/<name>/section_co_mentions")
@login_required
def api_section_co_mentions(name: str):
    """Create links between sections that mention the same entity."""

    if not get_dataset(name):
        abort(404)
    tid = dataset_operation_task.delay(
        name, "link_sections_by_entity", None, current_user.id
    ).id
    return jsonify({"task_id": tid})


@app.post("/api/datasets/<name>/author_org_links")
@login_required
def api_author_org_links(name: str):
    """Link document authors to their organizations."""
    if not get_dataset(name):
        abort(404)
    tid = dataset_operation_task.delay(
        name, "link_authors_organizations", None, current_user.id
    ).id
    return jsonify({"task_id": tid})


@app.get("/api/datasets/<name>/similar_chunks")
@login_required
def api_similar_chunks(name: str):
    """Return chunk IDs similar to ``cid``."""
    ds = get_dataset(name)
    if not ds:
        abort(404)
    cid = request.args.get("cid")
    k = int(request.args.get("k", 3))
    if not cid:
        return jsonify([])
    ids = ds.get_similar_chunks(cid, k=k)
    return jsonify(ids)


@app.get("/api/datasets/<name>/similar_chunks_data")
@login_required
def api_similar_chunks_data(name: str):
    """Return similar chunk info for ``cid``."""
    ds = get_dataset(name)
    if not ds:
        abort(404)
    cid = request.args.get("cid")
    k = int(request.args.get("k", 3))
    if not cid:
        return jsonify([])
    data = ds.get_similar_chunks_data(cid, k=k)
    return jsonify(data)


@app.get("/api/datasets/<name>/chunk_neighbors")
@login_required
def api_chunk_neighbors(name: str):
    """Return nearest neighbors for every chunk in the dataset."""

    ds = get_dataset(name)
    if not ds:
        abort(404)
    k = int(request.args.get("k", 3))
    neighbors = ds.get_chunk_neighbors(k=k)
    return jsonify(neighbors)


@app.get("/api/datasets/<name>/chunk_neighbors_data")
@login_required
def api_chunk_neighbors_data(name: str):
    """Return neighbor data for every chunk."""

    ds = get_dataset(name)
    if not ds:
        abort(404)
    k = int(request.args.get("k", 3))
    data = ds.get_chunk_neighbors_data(k=k)
    return jsonify(data)


@app.get("/api/datasets/<name>/similar_sections")
@login_required
def api_similar_sections(name: str):
    """Return section IDs similar to ``sid``."""

    ds = get_dataset(name)
    if not ds:
        abort(404)
    sid = request.args.get("sid")
    k = int(request.args.get("k", 3))
    if not sid:
        return jsonify([])
    ids = ds.get_similar_sections(sid, k=k)
    return jsonify(ids)


@app.get("/api/datasets/<name>/similar_documents")
@login_required
def api_similar_documents(name: str):
    """Return document IDs similar to ``did``."""

    ds = get_dataset(name)
    if not ds:
        abort(404)
    did = request.args.get("did")
    k = int(request.args.get("k", 3))
    if not did:
        return jsonify([])
    ids = ds.get_similar_documents(did, k=k)
    return jsonify(ids)


@app.get("/api/datasets/<name>/chunk_context")
@login_required
def api_chunk_context(name: str):
    """Return IDs of chunks surrounding ``cid``."""

    ds = get_dataset(name)
    if not ds:
        abort(404)
    cid = request.args.get("cid")
    before = int(request.args.get("before", 1))
    after = int(request.args.get("after", 1))
    if not cid:
        return jsonify([])
    ids = ds.get_chunk_context(cid, before=before, after=after)
    return jsonify(ids)


@app.get("/api/datasets/<name>/chunk_document")
@login_required
def api_chunk_document(name: str):
    """Return the document ID owning ``cid``."""

    ds = get_dataset(name)
    if not ds:
        abort(404)
    cid = request.args.get("cid")
    if not cid:
        return jsonify(None)
    doc = ds.get_document_for_chunk(cid)
    return jsonify(doc)


@app.get("/api/datasets/<name>/chunk_page")
@login_required
def api_chunk_page(name: str):
    """Return the page number for ``cid`` if available."""

    ds = get_dataset(name)
    if not ds:
        abort(404)
    cid = request.args.get("cid")
    if not cid:
        return jsonify(None)
    page = ds.get_page_for_chunk(cid)
    return jsonify(page)


@app.get("/api/datasets/<name>/section_document")
@login_required
def api_section_document(name: str):
    """Return the document ID owning ``sid``."""

    ds = get_dataset(name)
    if not ds:
        abort(404)
    sid = request.args.get("sid")
    if not sid:
        return jsonify(None)
    doc = ds.get_document_for_section(sid)
    return jsonify(doc)


@app.get("/api/datasets/<name>/section_page")
@login_required
def api_section_page(name: str):
    """Return the page number recorded for ``sid``."""

    ds = get_dataset(name)
    if not ds:
        abort(404)
    sid = request.args.get("sid")
    if not sid:
        return jsonify(None)
    page = ds.get_page_for_section(sid)
    return jsonify(page)


@app.get("/api/datasets/<name>/chunk_entities")
@login_required
def api_chunk_entities(name: str):
    ds = get_dataset(name)
    if not ds:
        abort(404)
    cid = request.args.get("cid")
    if not cid:
        return jsonify([])
    ids = ds.get_entities_for_chunk(cid)
    return jsonify(ids)


@app.get("/api/datasets/<name>/chunk_facts")
@login_required
def api_chunk_facts(name: str):
    ds = get_dataset(name)
    if not ds:
        abort(404)
    cid = request.args.get("cid")
    if not cid:
        return jsonify([])
    ids = ds.get_facts_for_chunk(cid)
    return jsonify(ids)


@app.get("/api/datasets/<name>/fact_sections")
@login_required
def api_fact_sections(name: str):
    ds = get_dataset(name)
    if not ds:
        abort(404)
    fid = request.args.get("fid")
    if not fid:
        return jsonify([])
    ids = ds.get_sections_for_fact(fid)
    return jsonify(ids)


@app.get("/api/datasets/<name>/fact_documents")
@login_required
def api_fact_documents(name: str):
    ds = get_dataset(name)
    if not ds:
        abort(404)
    fid = request.args.get("fid")
    if not fid:
        return jsonify([])
    ids = ds.get_documents_for_fact(fid)
    return jsonify(ids)


@app.get("/api/datasets/<name>/fact_pages")
@login_required
def api_fact_pages(name: str):
    """Return page numbers referencing ``fid``."""

    ds = get_dataset(name)
    if not ds:
        abort(404)
    fid = request.args.get("fid")
    if not fid:
        return jsonify([])
    pages = ds.get_pages_for_fact(fid)
    return jsonify(pages)


@app.get("/api/datasets/<name>/entity_documents")
@login_required
def api_entity_documents(name: str):
    ds = get_dataset(name)
    if not ds:
        abort(404)
    eid = request.args.get("eid")
    if not eid:
        return jsonify([])
    ids = ds.get_documents_for_entity(eid)
    return jsonify(ids)


@app.get("/api/datasets/<name>/entity_chunks")
@login_required
def api_entity_chunks(name: str):
    ds = get_dataset(name)
    if not ds:
        abort(404)
    eid = request.args.get("eid")
    if not eid:
        return jsonify([])
    ids = ds.get_chunks_for_entity(eid)
    return jsonify(ids)


@app.get("/api/datasets/<name>/entity_facts")
@login_required
def api_entity_facts(name: str):
    ds = get_dataset(name)
    if not ds:
        abort(404)
    eid = request.args.get("eid")
    if not eid:
        return jsonify([])
    ids = ds.get_facts_for_entity(eid)
    return jsonify(ids)


@app.get("/api/datasets/<name>/entity_pages")
@login_required
def api_entity_pages(name: str):
    """Return page numbers mentioning ``eid``."""

    ds = get_dataset(name)
    if not ds:
        abort(404)
    eid = request.args.get("eid")
    if not eid:
        return jsonify([])
    pages = ds.get_pages_for_entity(eid)
    return jsonify(pages)


@app.get("/api/datasets/<name>/search_hybrid")
@login_required
def api_search_hybrid(name: str):
    """Hybrid lexical/vector search on nodes."""
    ds = get_dataset(name)
    if not ds:
        abort(404)
    query = request.args.get("q")
    k = int(request.args.get("k", 5))
    node_type = request.args.get("type", "chunk")
    if not query:
        return jsonify([])
    ids = ds.search_hybrid(query, k=k, node_type=node_type)
    return jsonify(ids)


@app.get("/api/datasets/<name>/search_links")
@login_required
def api_search_links(name: str):
    """Search and expand results through graph links."""
    ds = get_dataset(name)
    if not ds:
        abort(404)
    query = request.args.get("q")
    k = int(request.args.get("k", 5))
    hops = int(request.args.get("hops", 1))
    if not query:
        return jsonify([])
    results = ds.search_with_links_data(query, k=k, hops=hops)
    return jsonify(results)


@app.get("/api/datasets/<name>/explain/<nid>")
@login_required
def api_explain_node(name: str, nid: str):
    """Return a 3-hop subgraph around ``nid`` with attention heatmap."""

    ds = get_dataset(name)
    if not ds:
        abort(404)
    hops = int(request.args.get("hops", 3))
    try:
        data = ds.explain_node(nid, hops=hops)
    except ValueError:
        abort(404)
    return jsonify(data)


@app.post("/api/datasets/<name>/consolidate")
@login_required
def api_consolidate(name: str):
    ds = get_dataset(name)
    if not ds:
        abort(404)
    ds.consolidate_schema()
    ds.history.append("Schema consolidated")
    save_dataset(ds)
    return jsonify({"message": "consolidated"})


@app.post("/api/datasets/<name>/communities")
@login_required
def api_communities(name: str):
    if not get_dataset(name):
        abort(404)
    tid = dataset_operation_task.delay(
        name, "detect_communities", None, current_user.id
    ).id
    return jsonify({"task_id": tid})


@app.post("/api/datasets/<name>/entity_groups")
@login_required
def api_entity_groups(name: str):
    if not get_dataset(name):
        abort(404)
    tid = dataset_operation_task.delay(
        name, "detect_entity_groups", None, current_user.id
    ).id
    return jsonify({"task_id": tid})


@app.post("/api/datasets/<name>/summaries")
@login_required
def api_summaries(name: str):
    if not get_dataset(name):
        abort(404)
    tid = dataset_operation_task.delay(
        name, "summarize_communities", None, current_user.id
    ).id
    return jsonify({"task_id": tid})


@app.post("/api/datasets/<name>/entity_group_summaries")
@login_required
def api_entity_group_summaries(name: str):
    if not get_dataset(name):
        abort(404)
    tid = dataset_operation_task.delay(
        name, "summarize_entity_groups", None, current_user.id
    ).id
    return jsonify({"task_id": tid})


@app.post("/api/datasets/<name>/resolve_entities")
@login_required
def api_resolve_entities(name: str):
    if not get_dataset(name):
        abort(404)
    data = request.get_json() or {}
    threshold = float(data.get("threshold", 0.8))
    aliases = data.get("aliases") if isinstance(data.get("aliases"), dict) else None
    tid = dataset_operation_task.delay(
        name,
        "resolve_entities",
        {"threshold": threshold, "aliases": aliases},
        current_user.id,
    ).id
    return jsonify({"task_id": tid})


@app.post("/api/datasets/<name>/predict_links")
@login_required
def api_predict_links(name: str):
    if not get_dataset(name):
        abort(404)
    use_graph = request.args.get("graph") == "true"
    tid = dataset_operation_task.delay(
        name,
        "predict_links",
        {"use_graph_embeddings": use_graph},
        current_user.id,
    ).id
    return jsonify({"task_id": tid})


@app.post("/api/datasets/<name>/enrich_entity/<eid>")
@login_required
def api_enrich_entity(name: str, eid: str):
    if not get_dataset(name):
        abort(404)
    tid = dataset_operation_task.delay(
        name,
        "enrich_entity",
        {"entity_id": eid},
        current_user.id,
    ).id
    return jsonify({"task_id": tid})


@app.post("/api/datasets/<name>/enrich_entity_dbpedia/<eid>")
@login_required
def api_enrich_entity_dbpedia(name: str, eid: str):
    if not get_dataset(name):
        abort(404)
    tid = dataset_operation_task.delay(
        name,
        "enrich_entity_dbpedia",
        {"entity_id": eid},
        current_user.id,
    ).id
    return jsonify({"task_id": tid})


@app.post("/api/datasets/<name>/trust")
@login_required
def api_trust(name: str):
    if not get_dataset(name):
        abort(404)
    tid = dataset_operation_task.delay(name, "score_trust", None, current_user.id).id
    return jsonify({"task_id": tid})


@app.post("/api/datasets/<name>/centrality")
@login_required
def api_centrality(name: str):
    """Compute centrality for dataset nodes."""

    if not get_dataset(name):
        abort(404)
    metric = request.args.get("metric", "degree")
    node_type = request.args.get("type", "entity")
    tid = dataset_operation_task.delay(
        name,
        "compute_centrality",
        {"node_type": node_type, "metric": metric},
        current_user.id,
    ).id
    return jsonify({"task_id": tid})


@app.post("/api/datasets/<name>/graph_embeddings")
@login_required
def api_graph_embeddings(name: str):
    """Generate Node2Vec embeddings for all nodes."""

    if not get_dataset(name):
        abort(404)

    data = request.get_json() or {}
    dims = int(data.get("dimensions", 64))
    walk = int(data.get("walk_length", 10))
    num = int(data.get("num_walks", 50))
    seed = int(data.get("seed", 0))
    workers = int(data.get("workers", 1))

    tid = dataset_operation_task.delay(
        name,
        "compute_graph_embeddings",
        {
            "dimensions": dims,
            "walk_length": walk,
            "num_walks": num,
            "seed": seed,
            "workers": workers,
        },
        current_user.id,
    ).id

    return jsonify({"task_id": tid})


@app.post("/api/datasets/<name>/extract_facts")
@login_required
def api_extract_facts(name: str):
    """Extract atomic facts from dataset chunks using an LLM."""
    if not get_dataset(name):
        abort(404)
    provider = request.json.get("provider") if request.json else None
    profile = request.json.get("profile") if request.json else None
    tid = dataset_extract_facts_task.delay(name, provider, profile, current_user.id).id
    return jsonify({"task_id": tid})


@app.post("/api/datasets/<name>/extract_entities")
@login_required
def api_extract_entities(name: str):
    """Run named entity recognition on dataset chunks."""

    if not get_dataset(name):
        abort(404)
    model = request.json.get("model") if request.json else "en_core_web_sm"
    tid = dataset_extract_entities_task.delay(name, model, current_user.id).id
    return jsonify({"task_id": tid})


@app.get("/api/datasets/<name>/conflicts")
@login_required
def api_conflicts(name: str):
    ds = get_dataset(name)
    if not ds:
        abort(404)
    conflicts = ds.find_conflicting_facts()
    return jsonify(conflicts)


@app.post("/api/datasets/<name>/mark_conflicts")
@login_required
def api_mark_conflicts(name: str):
    """Flag conflicting facts on graph edges."""

    if not get_dataset(name):
        abort(404)
    tid = dataset_operation_task.delay(
        name, "mark_conflicting_facts", None, current_user.id
    ).id
    return jsonify({"task_id": tid})


@app.post("/api/datasets/<name>/validate")
@login_required
def api_validate(name: str):
    """Run logical consistency checks on the dataset."""

    if not get_dataset(name):
        abort(404)
    tid = dataset_operation_task.delay(
        name, "validate_coherence", None, current_user.id
    ).id
    return jsonify({"task_id": tid})


@app.get("/api/datasets/<name>/export")
@login_required
def api_export_dataset(name: str):
    """Export the dataset asynchronously."""
    ds = get_dataset(name)
    if not ds:
        abort(404)
    fmt = ExportFormat(request.args.get("fmt", "jsonl"))
    tid = dataset_export_task.delay(name, fmt, current_user.id).id
    return jsonify({"task_id": tid})


@app.get("/api/datasets/<name>/export_result")
@login_required
def api_export_result(name: str):
    """Return previously exported dataset data from Redis."""

    if not REDIS:
        abort(404)
    fmt = request.args.get("fmt", "jsonl")
    key = f"dataset:{name}:export:{fmt}"
    data = REDIS.get(key)
    if data is None:
        abort(404)
    return (
        data,
        200,
        {"Content-Type": "application/json" if fmt == "json" else "text/plain"},
    )


@app.get("/api/datasets/<name>/history")
@login_required
def api_dataset_history(name: str):
    """Return the dataset's history events."""

    ds = get_dataset(name)
    if not ds:
        abort(404)
    events: list[dict[str, Any]] = []
    if REDIS:
        key = f"dataset:{name}:events"
        for raw in REDIS.lrange(key, 0, -1):
            if isinstance(raw, bytes):
                raw = raw.decode()
            try:
                events.append(json.loads(raw))
            except Exception:
                continue
    if not events:
        events = [asdict(e) for e in ds.events]
    return jsonify(events)


@app.get("/api/datasets/<name>/versions")
@login_required
def api_dataset_versions(name: str):
    """Return all generation versions for the dataset."""

    ds = get_dataset(name)
    if not ds:
        abort(404)
    return jsonify(ds.versions)


@app.get("/api/datasets/<name>/versions/<int:idx>")
@login_required
def api_dataset_version(name: str, idx: int):
    """Return a specific generation version."""

    ds = get_dataset(name)
    if not ds:
        abort(404)
    if idx < 1 or idx > len(ds.versions):
        abort(404)
    return jsonify(ds.versions[idx - 1])


@app.get("/api/datasets/<name>/progress")
@login_required
def api_dataset_progress(name: str):
    """Return generation progress stored in Redis."""

    if not get_dataset(name):
        abort(404)
    if not REDIS:
        return jsonify({})

    key = f"dataset:{name}:progress"
    progress: Dict[str, Any] = {}
    for k, v in REDIS.hgetall(key).items():
        if isinstance(k, bytes):
            k = k.decode()
        if isinstance(v, bytes):
            v = v.decode()
        try:
            progress[k] = json.loads(v)
        except Exception:
            progress[k] = v
    return jsonify(progress)


@app.post("/api/datasets/<name>/save_neo4j")
@login_required
def api_save_dataset_neo4j(name: str):
    """Persist the dataset graph to Neo4j asynchronously."""
    if not get_dataset(name):
        abort(404)
    if neo4j_breaker.current_state != 0:  # 0 == CLOSED
        abort(429)
    tid = dataset_save_neo4j_task.delay(name, current_user.id).id
    return jsonify({"task_id": tid})


@app.post("/api/datasets/<name>/load_neo4j")
@login_required
def api_load_dataset_neo4j(name: str):
    """Load the dataset graph from Neo4j asynchronously."""
    if not get_dataset(name):
        abort(404)
    tid = dataset_load_neo4j_task.delay(name, current_user.id).id
    return jsonify({"task_id": tid})


# ---------------------------------------------------------------------------
# Knowledge Graph API
# ---------------------------------------------------------------------------


@app.get("/api/graphs")
@login_required
def api_graphs():
    """Return list of knowledge graph names owned by the user."""
    names: set[str] = set()
    if REDIS:
        key = f"user:{current_user.id}:graphs"
        if REDIS.exists(key):
            for raw in REDIS.smembers(key):
                names.add(raw.decode() if isinstance(raw, bytes) else raw)
        else:
            for raw in REDIS.smembers("graphs"):
                name = raw.decode() if isinstance(raw, bytes) else raw
                try:
                    driver = get_neo4j_driver()
                    ds = DatasetBuilder.from_redis(REDIS, f"graph:{name}", driver)
                except KeyError:
                    continue
                if ds.owner_id in {None, current_user.id}:
                    names.add(name)

    return jsonify(sorted(names))


@app.post("/api/graphs")
@login_required
def api_create_graph():
    data = request.get_json() or {}
    name = data.get("name")
    docs = data.get("documents", [])
    if not name:
        abort(400)
    if name in GRAPHS:
        abort(400, description="Graph already exists")
    kg_ds = DatasetBuilder(DatasetType.TEXT, name=name)
    kg_ds.owner_id = current_user.id
    kg_ds.redis_client = REDIS
    kg_ds.neo4j_driver = get_neo4j_driver()
    for path in docs:
        ingest_into_dataset(path, kg_ds, config=config)
    kg_ds.history.append("Graph created")
    GRAPHS[name] = kg_ds
    save_graph(kg_ds)
    return jsonify({"message": "graph created"})


@app.get("/api/graphs/<name>")
@login_required
def api_graph_detail(name: str):
    ds = get_graph(name)
    if not ds:
        abort(404)
    graph = ds.graph.graph
    info = {
        "name": name,
        "num_nodes": len(graph.nodes),
        "num_edges": len(graph.edges),
    }
    return jsonify(info)


@app.get("/api/graphs/<name>/data")
@login_required
def api_graph_data(name: str):
    ds = get_graph(name)
    if not ds:
        abort(404)
    return jsonify(ds.graph.to_dict())


@app.get("/api/graphs/<name>/progress")
@login_required
def api_graph_progress(name: str):
    """Return progress information stored for the graph."""

    if not get_graph(name):
        abort(404)
    if not REDIS:
        return jsonify({})

    key = f"graph:{name}:progress"
    progress: Dict[str, Any] = {}
    for k, v in REDIS.hgetall(key).items():
        if isinstance(k, bytes):
            k = k.decode()
        if isinstance(v, bytes):
            v = v.decode()
        try:
            progress[k] = json.loads(v)
        except Exception:
            progress[k] = v
    return jsonify(progress)


@app.delete("/api/graphs/<name>")
@login_required
def api_delete_graph(name: str):
    """Delete a knowledge graph asynchronously."""
    if not get_graph(name):
        abort(404)
    GRAPHS.pop(name, None)
    tid = graph_delete_task.delay(name, current_user.id).id
    return jsonify({"task_id": tid})


@app.post("/api/graphs/<name>/save_neo4j")
@login_required
def api_graph_save_neo4j(name: str):
    """Persist the graph to Neo4j asynchronously."""
    if not get_graph(name):
        abort(404)
    if neo4j_breaker.current_state != 0:
        abort(429)
    tid = graph_save_neo4j_task.delay(name, current_user.id).id
    return jsonify({"task_id": tid})


@app.post("/api/graphs/<name>/load_neo4j")
@login_required
def api_graph_load_neo4j(name: str):
    """Load the graph from Neo4j asynchronously."""
    if not get_graph(name):
        abort(404)
    tid = graph_load_neo4j_task.delay(name, current_user.id).id
    return jsonify({"task_id": tid})


@app.route("/")
def index():
    """Serve the front-end application."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return app.send_static_file("index.html")
    return "Frontend not built", 200


@app.route("/datasets", methods=["GET", "POST"])
@login_required
def datasets():
    """List and create datasets"""
    form = DatasetForm()
    if form.validate_on_submit():
        name = form.name.data
        ds_type = DatasetType(form.dataset_type.data)
        DATASETS[name] = DatasetBuilder(ds_type, name=name)
        DATASETS[name].redis_client = REDIS
        DATASETS[name].history.append("Dataset created")
        save_dataset(DATASETS[name])
        flash("Dataset created", "success")
        return redirect(url_for("dataset_detail", name=name))
    return render_template("datasets.html", datasets=DATASETS, form=form)


@app.route("/datasets/<name>")
@login_required
def dataset_detail(name: str):
    ds = get_dataset(name)
    if not ds:
        abort(404)
    return render_template("dataset_detail.html", dataset=ds)


@app.get("/datasets/<name>/graph")
@login_required
def dataset_graph(name: str):
    """Return dataset knowledge graph as JSON."""
    ds = get_dataset(name)
    if not ds:
        abort(404)
    return jsonify(ds.graph.to_dict())


@app.get("/datasets/<name>/search")
@login_required
def dataset_search(name: str):
    """Return node ids matching the query."""
    ds = get_dataset(name)
    if not ds:
        abort(404)
    query = request.args.get("q")
    node_type = request.args.get("type", "chunk")
    if not query:
        return jsonify([])
    ids = ds.graph.search(query, node_type=node_type)
    return jsonify(ids)


@app.post("/datasets/<name>/ingest")
@login_required
def dataset_ingest(name: str):
    """Ingest a file or URL into the dataset knowledge graph."""
    ds = get_dataset(name)
    if not ds:
        abort(404)

    input_path = request.form.get("input_path")
    doc_id = request.form.get("doc_id") or None
    high_res = bool(request.form.get("high_res"))
    ocr = bool(request.form.get("ocr"))
    if not input_path:
        flash("Input path is required", "warning")
        return redirect(url_for("dataset_detail", name=name))

    try:
        ds.ingest_file(
            input_path,
            doc_id=doc_id,
            config=config,
            high_res=high_res,
            ocr=ocr,
        )
        flash("Document ingested", "success")
    except Exception as e:  # pragma: no cover - flash message only
        flash(f"Error ingesting document: {e}", "danger")

    return redirect(url_for("dataset_detail", name=name))


@app.post("/datasets/<name>/save_neo4j")
@login_required
def save_dataset_neo4j(name: str):
    """Persist the dataset graph to Neo4j."""
    ds = get_dataset(name)
    if not ds:
        abort(404)
    driver = get_neo4j_driver()
    if not driver:
        abort(500, description="Neo4j not configured")
    ds.graph.to_neo4j(driver, dataset=name)
    driver.close()
    ds.history.append("Saved to Neo4j")
    save_dataset(ds)
    flash("Graph saved to Neo4j", "success")
    return redirect(url_for("dataset_detail", name=name))


@app.post("/datasets/<name>/load_neo4j")
@login_required
def load_dataset_neo4j(name: str):
    """Load the dataset graph from Neo4j."""
    ds = get_dataset(name)
    if not ds:
        abort(404)
    driver = get_neo4j_driver()
    if not driver:
        abort(500, description="Neo4j not configured")
    ds.graph = KnowledgeGraph.from_neo4j(driver, dataset=name)
    driver.close()
    ds.history.append("Loaded from Neo4j")
    save_dataset(ds)
    flash("Graph loaded from Neo4j", "success")
    return redirect(url_for("dataset_detail", name=name))


@app.post("/datasets/<name>/delete")
@login_required
def delete_dataset(name: str):
    ds = get_dataset(name)
    if not ds:
        abort(404)
    DATASETS.pop(name, None)
    dataset_delete_task.delay(name, current_user.id)
    flash("Dataset deletion started", "success")
    return redirect(url_for("datasets"))


@app.post("/datasets/<name>/copy")
@login_required
def copy_dataset(name: str):
    ds = get_dataset(name)
    if not ds:
        abort(404)
    new_name = f"{name}_copy"
    counter = 1
    while new_name in DATASETS:
        counter += 1
        new_name = f"{name}_copy{counter}"
    DATASETS[new_name] = ds.clone(name=new_name)
    DATASETS[new_name].history.append(f"Copied from {name}")
    save_dataset(DATASETS[new_name])
    flash("Dataset copied", "success")
    return redirect(url_for("dataset_detail", name=new_name))


@app.route("/create", methods=["GET", "POST"])
@login_required
def create():
    """Create content from text"""
    form = CreateForm()
    default_provider = get_llm_provider(config)

    if not form.provider.data:
        form.provider.data = default_provider

    provider = form.provider.data or default_provider

    if form.validate_on_submit():
        try:
            input_file = form.input_file.data
            content_type = form.content_type.data
            num_pairs = form.num_pairs.data
            model = form.model.data or None
            api_base = form.api_base.data or None

            result = process_file(
                file_path=input_file,
                content_type=content_type,
                num_pairs=num_pairs,
                provider=provider,
                api_base=api_base,
                model=model,
                config_path=None,  # Use default config
                verbose=True,
            )

            content_type_labels = {
                "qa": "QA pairs",
                "summary": "summary",
                "cot": "Chain of Thought examples",
                "cot-enhance": "CoT enhanced conversation",
            }
            content_label = content_type_labels.get(content_type, content_type)

            flash(f"Successfully generated {content_label}!", "success")
            return redirect(url_for("datasets"))

        except Exception as e:
            flash(f"Error: {str(e)}", "danger")

    # Get the list of available input files
    input_files: list[str] = []

    return render_template(
        "create.html", form=form, provider=provider, input_files=input_files
    )


@app.route("/curate", methods=["GET", "POST"])
@login_required
def curate():
    """Curate QA pairs interface"""
    form = CurateForm()
    default_provider = get_llm_provider(config)

    if not form.provider.data:
        form.provider.data = default_provider

    provider = form.provider.data or default_provider

    if form.validate_on_submit():
        try:
            input_file = form.input_file.data
            model = form.model.data or None
            api_base = form.api_base.data or None

            result = curate_qa_pairs(
                input_path=input_file,
                provider=provider,
                api_base=api_base,
                model=model,
                config_path=None,  # Use default config
                verbose=True,
                async_mode=False,
            )

            flash("Successfully curated QA pairs!", "success")
            return redirect(url_for("datasets"))

        except Exception as e:
            flash(f"Error: {str(e)}", "danger")

    # Get the list of available JSON files
    json_files: list[str] = []

    return render_template(
        "curate.html", form=form, provider=provider, json_files=json_files
    )


@app.route("/ingest", methods=["GET", "POST"])
@login_required
def ingest():
    """Ingest and parse documents"""
    form = IngestForm()

    if form.validate_on_submit():
        try:
            input_type = form.input_type.data

            if input_type == "file":
                # Handle file upload
                if not form.upload_file.data:
                    flash("Please upload a file", "warning")
                    return render_template("ingest.html", form=form)

                temp_file = form.upload_file.data
                file_extension = Path(temp_file.filename).suffix

                import tempfile

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=file_extension
                ) as tmp:
                    temp_file.save(tmp.name)
                    input_path = tmp.name
            else:
                # URL or local path
                input_path = form.input_path.data
                if not input_path:
                    flash("Please enter a valid path or URL", "warning")
                    return render_template("ingest.html", form=form)

            # Process the file or URL
            ingest_process_file(
                file_path=input_path,
                config=config,
            )

            # Clean up temporary file if it was an upload
            if input_type == "file":
                try:
                    os.unlink(input_path)
                except Exception:
                    pass

            flash("Successfully parsed document!", "success")
            return redirect(url_for("datasets"))

        except Exception as e:
            flash(f"Error: {str(e)}", "danger")

    # Get some example URLs for different document types
    examples = {
        "PDF": "path/to/document.pdf",
        "YouTube": "https://www.youtube.com/watch?v=example",
        "Web Page": "https://example.com/article",
        "Word Document": "path/to/document.docx",
        "PowerPoint": "path/to/presentation.pptx",
        "Text File": "path/to/document.txt",
    }

    return render_template("ingest.html", form=form, examples=examples)


def run_server(host="127.0.0.1", port=5000, debug=False):
    """Run the Flask server."""

    if REDIS:
        load_datasets_from_redis()
        load_graphs_from_redis()

    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    run_server(host=host, port=port, debug=debug)
