"""Task orchestration helpers for ingestion and export pipelines."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import redis

try:  # optional dependency
    from celery import Celery
except Exception:  # pragma: no cover - optional dependency missing

    class _AsyncResult:
        """Simple stand-in for :class:`celery.result.AsyncResult`."""

        def __init__(self, result):
            self._result = result

        def get(self, timeout: float | None = None):
            return self._result

    class Celery:  # type: ignore[misc]
        """Very small stub of :class:`Celery` used when the dependency is missing."""

        def __init__(self, *a, **k):
            """Create a minimal Celery substitute."""

            from types import SimpleNamespace

            self.conf = SimpleNamespace()

        def task(self, fn=None, **opts):
            def decorator(func):
                def delay(*args, **kwargs):  # type: ignore[misc]
                    return _AsyncResult(func(*args, **kwargs))

                func.delay = delay

                # mimic Celery's ``apply_async`` for tests that expect it
                def apply_async(args=None, kwargs=None, **_opts):  # type: ignore[misc]
                    return delay(*(args or []), **(kwargs or {}))

                func.apply_async = apply_async
                return func

            return decorator(fn) if fn else decorator


from pydantic import ValidationError

from datacreek.analysis.monitoring import (
    ingest_rate_limited_total,
    ingest_total,
    ingest_validation_fail_total,
)
from datacreek.backends import (
    get_neo4j_driver,
    get_redis_client,
    get_redis_graph,
    get_s3_storage,
)
from datacreek.core.create import process_file as generate_data
from datacreek.core.curate import curate_qa_pairs
from datacreek.core.dataset import DatasetBuilder
from datacreek.core.ingest import IngestOptions, IngestOptionsModel
from datacreek.core.ingest import process_file as ingest_file
from datacreek.core.knowledge_graph import KnowledgeGraph
from datacreek.core.save_as import convert_format
from datacreek.db import Dataset, SessionLocal, SourceData
from datacreek.models import LLMService
from datacreek.models.export_format import ExportFormat
from datacreek.models.llm_client import LLMClient
from datacreek.models.task_status import TaskStatus
from datacreek.pipelines import GenerationOptionsModel
from datacreek.schemas import DatasetName
from datacreek.services import create_dataset, create_source
from datacreek.utils import backpressure, consume_token, export_delta
from datacreek.utils import extract_entities as extract_entities_func
from datacreek.utils import extract_facts as extract_facts_func
from datacreek.utils import lakefs_commit
from datacreek.utils.config import load_config_with_overrides
from datacreek.utils.neo4j_breaker import CircuitBreakerError, neo4j_breaker
from schemas import AudioIngest, BaseIngest, ImageIngest, PdfIngest

CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "memory://")
CELERY_BACKEND_URL = os.environ.get("CELERY_RESULT_BACKEND", "cache+memory://")

celery_app = Celery("datacreek", broker=CELERY_BROKER_URL, backend=CELERY_BACKEND_URL)
celery_app.conf.accept_content = ["json"]
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
celery_app.conf.task_always_eager = os.environ.get(
    "CELERY_TASK_ALWAYS_EAGER", "false"
).lower() in {
    "1",
    "true",
}
celery_app.conf.task_store_eager_result = True

logger = logging.getLogger(__name__)


def _update_status(
    client: redis.Redis,
    key: str,
    status: TaskStatus | str,
    progress: float | None = None,
) -> None:
    """Set progress status information and record the timeline."""

    val = status.value if isinstance(status, TaskStatus) else status
    entry = {"status": val}
    if progress is not None:
        entry["progress"] = progress
    entry["time"] = datetime.now(timezone.utc).isoformat()
    try:
        client.hset(key, "status", val)
        if progress is not None:
            client.hset(key, "progress", progress)
        client.rpush(f"{key}:history", json.dumps(entry))
    except Exception:
        logger.exception("Failed to update task status for %s", key)


def _record_error(
    client: redis.Redis,
    key: str,
    exc: Exception,
    dataset: DatasetBuilder | None = None,
) -> None:
    """Save task failure details in ``client`` under ``key`` and log event."""
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    entry = {
        "status": TaskStatus.FAILED.value,
        "error": str(exc),
        "exc_type": type(exc).__name__,
        "traceback": tb,
        "time": datetime.now(timezone.utc).isoformat(),
    }
    try:
        client.hset(
            key,
            mapping={
                "error": entry["error"],
                "exc_type": entry["exc_type"],
                "traceback": entry["traceback"],
                "status": TaskStatus.FAILED.value,
            },
        )
        client.rpush(f"{key}:history", json.dumps(entry))
    except Exception:
        logger.exception("Failed to record error for %s", key)

    if dataset is not None:
        try:
            dataset._record_event(
                "task_error",
                f"{type(exc).__name__}: {exc}",
                traceback=tb,
            )
        except Exception:
            logger.exception("Failed to record dataset error event for %s", key)


@celery_app.task
def ingest_task(  # pragma: no cover - heavy
    user_id: int,
    path: str,
    *,
    high_res: bool = False,
    ocr: bool = False,
    use_unstructured: bool | None = None,
    extract_entities: bool = False,
    extract_facts: bool = False,
) -> dict:
    with SessionLocal() as db:
        content = ingest_file(
            path, high_res=high_res, ocr=ocr, use_unstructured=use_unstructured
        )
        ents = extract_entities_func(content) if extract_entities else None
        facts = extract_facts_func(content) if extract_facts else None
        src = create_source(
            db,
            user_id,
            path,
            content,
            entities=ents,
            facts=facts,
        )
        return {"id": src.id}


@celery_app.task
def generate_task(  # pragma: no cover - heavy
    user_id: int,
    src_id: int,
    content_type: str,
    num_pairs: int | None,
    *,
    provider: str | None = None,
    profile: str | None = None,
    model: str | None = None,
    api_base: str | None = None,
    config_path: str | None = None,
    generation: dict | None = None,
    prompts: dict | None = None,
) -> dict:
    with SessionLocal() as db:
        src = db.get(SourceData, src_id)
        if not src or src.owner_id != user_id:
            raise RuntimeError("Source not found")
        from datacreek.utils.config import load_config_with_overrides

        overrides = {}
        if generation is not None:
            overrides["generation"] = generation
        if prompts is not None:
            overrides["prompts"] = prompts
        load_path = str(config_path) if config_path else None
        load_overrides = overrides if overrides else None
        load_config_with_overrides(load_path, load_overrides)
        out = generate_data(
            None,
            Path(config_path) if config_path else None,
            api_base,
            model,
            content_type,
            num_pairs,
            False,
            provider=provider,
            profile=profile,
            kg=KnowledgeGraph(),
            document_text=src.content,
            config_overrides=overrides if overrides else None,
        )
        ds = create_dataset(db, user_id, src_id, content=json.dumps(out))
        return {"id": ds.id}


@celery_app.task
def curate_task(
    user_id: int, ds_id: int, threshold: float | None
) -> dict:  # pragma: no cover - heavy
    with SessionLocal() as db:
        ds = db.get(Dataset, ds_id)
        if not ds or ds.owner_id != user_id:
            raise RuntimeError("Dataset not found")
        data = json.loads(ds.content or "{}")
        result = curate_qa_pairs(
            data, threshold, None, None, None, False, async_mode=False
        )
        ds.content = json.dumps(result)
        db.commit()
        db.refresh(ds)
        return {"id": ds.id}


@celery_app.task
def save_task(
    user_id: int, ds_id: int, fmt: ExportFormat
) -> dict:  # pragma: no cover - heavy
    with SessionLocal() as db:
        ds = db.get(Dataset, ds_id)
        if not ds or ds.owner_id != user_id:
            raise RuntimeError("Dataset not found")
        data = json.loads(ds.content or "{}")
        if isinstance(fmt, str):
            fmt = ExportFormat(fmt)
        out = convert_format(data, fmt.value, {}, "json")
        ds.content = out if isinstance(out, str) else json.dumps(out)
        db.commit()
        db.refresh(ds)
        return {"id": ds.id}


@celery_app.task
def dataset_ingest_task(  # pragma: no cover - heavy
    name: DatasetName, path: str, user_id: int | None = None, **kwargs
) -> dict:
    """Ingest a file into a persisted dataset."""
    if not backpressure.acquire_slot_with_backoff(
        int(os.getenv("INGEST_BACKOFF_RETRIES", "3")),
        float(os.getenv("INGEST_BACKOFF_BASE", "0.05")),
        spool_dir=os.getenv("INGEST_SPOOL_DIR"),
        spool_data={"name": name, "path": path},
    ):
        raise RuntimeError("ingest queue full")
    client = get_redis_client()
    tenant_id = str(user_id) if user_id is not None else "public"
    if not consume_token(tenant_id, client=client):
        if ingest_rate_limited_total is not None:
            ingest_rate_limited_total.inc()
        backpressure.release_slot()
        raise RuntimeError("rate limit exceeded")
    if ingest_total is not None:
        ingest_total.inc()
    driver = get_neo4j_driver()
    storage = get_s3_storage()
    ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    ds.redis_client = client
    opt_fields = IngestOptionsModel.model_fields.keys()
    opt_args = {k: kwargs.pop(k) for k in list(kwargs) if k in opt_fields}
    options = IngestOptionsModel(**opt_args).to_options() if opt_args else None
    async_mode = kwargs.pop("async_mode", False)

    # Validate payload according to resource type
    ext = Path(path).suffix.lower()
    try:
        if ext in {".jpg", ".jpeg", ".png", ".gif"}:
            if not {"width", "height", "blur_score"} <= opt_args.keys():
                try:
                    from datacreek.utils.quality_metrics import blur_score as _bs
                    from datacreek.utils.quality_metrics import image_dimensions

                    w, h = image_dimensions(path)
                    opt_args.setdefault("width", w)
                    opt_args.setdefault("height", h)
                    opt_args.setdefault("blur_score", _bs(path))
                except Exception:
                    pass
            ImageIngest(path=path, **opt_args)
        elif ext in {".wav", ".mp3", ".flac"}:
            if not {"sample_rate", "duration", "snr"} <= opt_args.keys():
                try:
                    from datacreek.utils.quality_metrics import audio_metrics

                    sr, dur, snr = audio_metrics(path)
                    opt_args.setdefault("sample_rate", sr)
                    opt_args.setdefault("duration", dur)
                    opt_args.setdefault("snr", snr)
                except Exception:
                    pass
            AudioIngest(path=path, **opt_args)
        elif ext == ".pdf":
            if "entropy" not in opt_args:
                try:
                    from datacreek.utils.quality_metrics import text_entropy

                    with open(path, "rb") as fh:
                        text = fh.read().decode("latin1", "ignore")
                    opt_args["entropy"] = text_entropy(text)
                except Exception:
                    pass
            PdfIngest(path=path, **opt_args)
        else:
            BaseIngest(path=path)
    except ValidationError:
        if ingest_validation_fail_total is not None:
            ingest_validation_fail_total.inc()
        raise
    key = f"dataset:{name}:progress"
    start_ts = datetime.now(timezone.utc).isoformat()
    client.hset(key, "ingest_start", start_ts)
    _update_status(client, key, TaskStatus.INGESTING, 0.0)
    client.hset(key, "ingested_chunks", 0)
    if opt_args:
        client.hset(key, "ingestion_params", json.dumps(opt_args))

    def chunk_progress(_idx: int) -> None:
        try:
            client.hincrby(key, "ingested_chunks", 1)
        except Exception:
            logger.exception("Failed to record chunk progress")

    try:
        if async_mode:
            doc_id = asyncio.run(
                ds.ingest_file_async(
                    path,
                    options=options,
                    progress_callback=chunk_progress,
                    **kwargs,
                )
            )
        else:
            doc_id = ds.ingest_file(
                path,
                options=options,
                progress_callback=chunk_progress,
                **kwargs,
            )
        _update_status(client, key, TaskStatus.INGESTING, 0.5)
        client.hincrby(key, "ingested", 1)
        ts = datetime.now(timezone.utc).isoformat()
        client.hset(
            key,
            f"ingested:{doc_id}",
            json.dumps({"path": path, "time": ts}),
        )
        client.hset(
            key,
            "last_ingested",
            json.dumps({"id": doc_id, "path": path, "time": ts}),
        )
        client.hset(key, "ingest_finish", ts)
        _update_status(client, key, TaskStatus.COMPLETED, 1.0)
        return {"stage": ds.stage, "events": len(ds.events)}
    except Exception as exc:
        _record_error(client, key, exc, ds)
        raise
    finally:
        backpressure.release_slot()
        if driver:
            driver.close()


def enqueue_dataset_ingest(
    name: DatasetName, path: str, user_id: int | None = None, **kwargs
) -> object:
    """Enqueue ingestion in Kafka if configured, else run via Celery."""

    if os.getenv("KAFKA_BOOTSTRAP_SERVERS"):
        from datacreek.utils.kafka_queue import enqueue_ingest

        return enqueue_ingest(name, path, user_id=user_id, **kwargs)

    return dataset_ingest_task.apply_async(args=[name, path, user_id], kwargs=kwargs)


@celery_app.task
def dataset_generate_task(  # pragma: no cover - heavy
    name: DatasetName,
    params: dict | None = None,
    user_id: int | None = None,
    *,
    provider: str | None = None,
    profile: str | None = None,
    api_base: str | None = None,
    model: str | None = None,
) -> dict:
    """Run the post-KG pipeline for a persisted dataset."""
    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    ds.redis_client = client
    params = params or {}
    opt_fields = GenerationOptionsModel.model_fields.keys()
    opt_args = {k: params.pop(k) for k in list(params) if k in opt_fields}
    options = GenerationOptionsModel(**opt_args).to_options() if opt_args else None
    key = f"dataset:{name}:progress"
    ts = datetime.now(timezone.utc).isoformat()
    client.hset(key, "generate_start", ts)
    _update_status(client, key, TaskStatus.GENERATING, 0.0)
    if opt_args or params:
        all_args = opt_args.copy()
        all_args.update(params)
        client.hset(key, "generation_params", json.dumps(all_args))
    opt_dict = asdict(options) if is_dataclass(options) else {}
    if provider or profile or api_base or model:
        ds.configure_llm_service(
            provider=provider or "vllm",
            profile=profile,
            api_base=api_base,
            model=model,
        )
    try:
        ds.run_post_kg_pipeline(redis_client=client, **opt_dict, **params)
        try:
            ds._enforce_policy([1])
        except Exception:
            logger.exception("Invariant enforcement failed")
        end_ts = datetime.now(timezone.utc).isoformat()
        client.hset(key, "generated_version", json.dumps(len(ds.versions)))
        client.hset(key, "generate_finish", end_ts)
        _update_status(client, key, TaskStatus.COMPLETED, 1.0)
        return {"stage": ds.stage, "versions": len(ds.versions)}
    except Exception as exc:
        _record_error(client, key, exc, ds)
        raise


@celery_app.task
def dataset_cleanup_task(  # pragma: no cover - heavy
    name: str, params: dict | None = None, user_id: int | None = None
) -> dict:
    """Run cleanup operations on a persisted dataset."""

    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    ds.redis_client = client
    params = params or {}
    key = f"dataset:{name}:progress"
    start_ts = datetime.now(timezone.utc).isoformat()
    client.hset(key, "cleanup_start", start_ts)
    _update_status(client, key, TaskStatus.CLEANUP, 0.0)
    if params:
        client.hset(key, "cleanup_params", json.dumps(params))

    try:
        from datacreek.core.knowledge_graph import verify_thresholds

        verify_thresholds()
        removed, cleaned = ds.cleanup_graph(**params)
        _update_status(client, key, TaskStatus.CLEANUP, 0.5)
        try:
            ds.monitor_and_remediate([1])
        except Exception:
            logger.exception("Policy remediation failed")

        ts = datetime.now(timezone.utc).isoformat()
        client.hset(
            key,
            "cleanup",
            json.dumps({"removed": removed, "cleaned": cleaned, "time": ts}),
        )
        client.hset(key, "cleanup_finish", ts)
        _update_status(client, key, TaskStatus.COMPLETED, 1.0)

        return {"stage": ds.stage, "removed": removed, "cleaned": cleaned}
    except Exception as exc:
        _record_error(client, key, exc, ds)
        raise


@celery_app.task
def dataset_export_task(  # pragma: no cover - heavy
    name: DatasetName,
    fmt: ExportFormat = ExportFormat.JSONL,
    user_id: int | None = None,
) -> dict:
    """Format the latest generation result and mark the dataset exported."""

    if isinstance(fmt, str):
        fmt = ExportFormat(fmt)
    client = get_redis_client()
    driver = get_neo4j_driver()
    storage = get_s3_storage()
    ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    ds.redis_client = client
    data = None
    key: str
    progress_key = f"dataset:{name}:progress"
    _update_status(client, progress_key, TaskStatus.EXPORTING, 0.0)
    try:
        commit_id = None
        if ds.versions and ds.versions[-1].get("result") is not None:
            data = ds.versions[-1]["result"]
            if fmt is ExportFormat.DELTA:
                root_dir = os.getenv("DELTA_EXPORT_ROOT", "delta_exports")
                path = export_delta(
                    data,
                    root=root_dir,
                    org_id=ds.owner_id or "anon",
                    kind=ds.dataset_type.value,
                )
                repo = os.getenv("LAKEFS_REPO")
                commit_id = lakefs_commit(path, repo) if repo else None
                delta_optimize(Path(root_dir))
                delta_vacuum(Path(root_dir))
                _update_status(client, progress_key, TaskStatus.EXPORTING, 0.5)
                key = str(path)
                payload = ""
            else:
                formatted = convert_format(data, fmt.value, {}, "json")
                _update_status(client, progress_key, TaskStatus.EXPORTING, 0.5)
                key = f"dataset:{name}:export:{fmt.value}"
                payload = (
                    formatted if isinstance(formatted, str) else json.dumps(formatted)
                )
        else:
            key = f"dataset:{name}:export:json"
            payload = json.dumps(ds.to_dict())
        if payload:
            client.set(key, payload)
            s3_key = storage.save(key, payload) if storage else None
        else:
            s3_key = None

        ds.mark_exported()
        ts = datetime.now(timezone.utc).isoformat()
        info = {"fmt": fmt.value, "key": key, "time": ts}
        if commit_id:
            info["lakefs_commit"] = commit_id
        if storage:
            info["s3_key"] = s3_key
        client.hset(progress_key, "export", json.dumps(info))
        _update_status(client, progress_key, TaskStatus.COMPLETED, 1.0)

        return {"stage": ds.stage, "key": key}
    except Exception as exc:
        _record_error(client, progress_key, exc, ds)
        raise


@celery_app.task
def dataset_save_neo4j_task(
    name: DatasetName, user_id: int | None = None
) -> dict:  # pragma: no cover - heavy
    """Persist the dataset graph to Neo4j."""

    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    ds.redis_client = client
    if not driver:
        raise RuntimeError("Neo4j not configured")
    key = f"dataset:{name}:progress"
    start_ts = datetime.now(timezone.utc).isoformat()
    client.hset(key, "save_neo4j_start", start_ts)
    _update_status(client, key, TaskStatus.SAVING_NEO4J, 0.0)
    try:
        neo4j_breaker.call(ds.save_neo4j, driver)
        _update_status(client, key, TaskStatus.SAVING_NEO4J, 0.5)
        driver.close()
        ds.redis_client = client
        ts = datetime.now(timezone.utc).isoformat()
        client.hset(
            key,
            "save_neo4j",
            json.dumps({"nodes": len(ds.graph.graph), "time": ts}),
        )
        client.hset(key, "save_neo4j_finish", ts)
        _update_status(client, key, TaskStatus.COMPLETED, 1.0)
        return {"stage": ds.stage}
    except CircuitBreakerError as exc:
        _record_error(client, key, exc, ds)
        raise RuntimeError("neo4j circuit open") from exc
    except Exception as exc:
        _record_error(client, key, exc, ds)
        raise


@celery_app.task
def dataset_load_neo4j_task(
    name: DatasetName, user_id: int | None = None
) -> dict:  # pragma: no cover - heavy
    """Load the dataset graph from Neo4j."""

    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    ds.redis_client = client
    if not driver:
        raise RuntimeError("Neo4j not configured")
    key = f"dataset:{name}:progress"
    start_ts = datetime.now(timezone.utc).isoformat()
    client.hset(key, "load_neo4j_start", start_ts)
    _update_status(client, key, TaskStatus.LOADING_NEO4J, 0.0)
    try:
        ds.load_neo4j(driver)
        _update_status(client, key, TaskStatus.LOADING_NEO4J, 0.5)
        driver.close()
        ds.neo4j_driver = None
        ds.redis_client = client
        ts = datetime.now(timezone.utc).isoformat()
        client.hset(
            key,
            "load_neo4j",
            json.dumps({"nodes": len(ds.graph.graph), "time": ts}),
        )
        client.hset(key, "load_neo4j_finish", ts)
        _update_status(client, key, TaskStatus.COMPLETED, 1.0)
        return {"nodes": len(ds.graph.graph)}
    except Exception as exc:
        _record_error(client, key, exc, ds)
        raise


@celery_app.task
def dataset_save_redis_graph_task(  # pragma: no cover - heavy
    name: DatasetName, user_id: int | None = None
) -> dict:
    """Persist the dataset graph to RedisGraph."""

    client = get_redis_client()
    ds = DatasetBuilder.from_redis(client, f"dataset:{name}", None)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    ds.redis_client = client
    graph = get_redis_graph(name)
    if graph is None:
        raise RuntimeError("RedisGraph not configured")
    key = f"dataset:{name}:progress"
    start_ts = datetime.now(timezone.utc).isoformat()
    client.hset(key, "save_redis_graph_start", start_ts)
    _update_status(client, key, TaskStatus.SAVING_REDIS_GRAPH, 0.0)
    try:
        ds.save_redis_graph(graph)
        _update_status(client, key, TaskStatus.SAVING_REDIS_GRAPH, 0.5)
        ts = datetime.now(timezone.utc).isoformat()
        client.hset(
            key,
            "save_redis_graph",
            json.dumps({"nodes": len(ds.graph.graph), "time": ts}),
        )
        client.hset(key, "save_redis_graph_finish", ts)
        _update_status(client, key, TaskStatus.COMPLETED, 1.0)
        return {"nodes": len(ds.graph.graph)}
    except Exception as exc:
        _record_error(client, key, exc, ds)
        raise


@celery_app.task
def dataset_load_redis_graph_task(  # pragma: no cover - heavy
    name: DatasetName, user_id: int | None = None
) -> dict:
    """Load the dataset graph from RedisGraph."""

    client = get_redis_client()
    ds = DatasetBuilder.from_redis(client, f"dataset:{name}", None)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    ds.redis_client = client
    graph = get_redis_graph(name)
    if graph is None:
        raise RuntimeError("RedisGraph not configured")
    key = f"dataset:{name}:progress"
    start_ts = datetime.now(timezone.utc).isoformat()
    client.hset(key, "load_redis_graph_start", start_ts)
    _update_status(client, key, TaskStatus.LOADING_REDIS_GRAPH, 0.0)
    try:
        ds.load_redis_graph(graph)
        _update_status(client, key, TaskStatus.LOADING_REDIS_GRAPH, 0.5)
        ts = datetime.now(timezone.utc).isoformat()
        client.hset(
            key,
            "load_redis_graph",
            json.dumps({"nodes": len(ds.graph.graph), "time": ts}),
        )
        client.hset(key, "load_redis_graph_finish", ts)
        _update_status(client, key, TaskStatus.COMPLETED, 1.0)
        return {"nodes": len(ds.graph.graph)}
    except Exception as exc:
        _record_error(client, key, exc, ds)
        raise


@celery_app.task
def dataset_operation_task(  # pragma: no cover - heavy
    name: DatasetName,
    operation: str,
    params: dict | None = None,
    user_id: int | None = None,
) -> dict:
    """Run an arbitrary dataset method and persist the result."""

    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    ds.redis_client = client
    params = params or {}
    func = getattr(ds, operation)
    prog_key = f"dataset:{name}:progress"
    start_ts = datetime.now(timezone.utc).isoformat()
    client.hset(prog_key, f"{operation}_start", start_ts)
    client.hset(prog_key, "operation", operation)
    _update_status(client, prog_key, TaskStatus.OPERATION, 0.0)
    try:
        result = func(**params)
        _update_status(client, prog_key, TaskStatus.OPERATION, 0.5)
        ts = datetime.now(timezone.utc).isoformat()
        try:
            client.hset(
                prog_key,
                operation,
                json.dumps({"result": result, "time": ts}, default=str),
            )
        except Exception:
            client.hset(prog_key, operation, json.dumps({"time": ts}))
        _update_status(client, prog_key, TaskStatus.COMPLETED, 1.0)
        return {"result": result, "stage": ds.stage}
    except Exception as exc:
        _record_error(client, prog_key, exc, ds)
        raise
    finally:
        if driver:
            driver.close()


@celery_app.task
def dataset_prune_versions_task(  # pragma: no cover - heavy
    name: DatasetName, limit: int | None = None, user_id: int | None = None
) -> dict:
    """Prune stored versions for ``name`` down to ``limit``."""

    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    ds.redis_client = client
    key = f"dataset:{name}:progress"
    start_ts = datetime.now(timezone.utc).isoformat()
    client.hset(key, "prune_versions_start", start_ts)
    client.hset(key, "operation", "prune_versions")
    _update_status(client, key, TaskStatus.OPERATION, 0.0)
    try:
        removed = ds.prune_versions(limit)
        ts = datetime.now(timezone.utc).isoformat()
        client.hset(key, "prune_versions", json.dumps({"removed": removed, "time": ts}))
        client.hset(key, "prune_versions_finish", ts)
        _update_status(client, key, TaskStatus.COMPLETED, 1.0)
        return {"removed": removed, "versions": len(ds.versions)}
    except Exception as exc:
        _record_error(client, key, exc, ds)
        raise
    finally:
        if driver:
            driver.close()


@celery_app.task
def dataset_restore_version_task(  # pragma: no cover - heavy
    name: DatasetName, index: int, user_id: int | None = None
) -> dict:
    """Restore ``index`` for ``name`` as the latest dataset version."""

    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    ds.redis_client = client
    key = f"dataset:{name}:progress"
    start_ts = datetime.now(timezone.utc).isoformat()
    client.hset(key, "restore_version_start", start_ts)
    client.hset(key, "operation", "restore_version")
    _update_status(client, key, TaskStatus.OPERATION, 0.0)
    try:
        ds.restore_version(index)
        ts = datetime.now(timezone.utc).isoformat()
        client.hset(
            key,
            "restore_version",
            json.dumps({"index": index, "time": ts}),
        )
        client.hset(key, "restore_version_finish", ts)
        _update_status(client, key, TaskStatus.COMPLETED, 1.0)
        return {"versions": len(ds.versions)}
    except Exception as exc:
        _record_error(client, key, exc, ds)
        raise
    finally:
        if driver:
            driver.close()


@celery_app.task
def dataset_delete_version_task(  # pragma: no cover - heavy
    name: DatasetName, index: int, user_id: int | None = None
) -> dict:
    """Delete ``index`` from ``name`` and persist the dataset."""

    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    ds.redis_client = client
    key = f"dataset:{name}:progress"
    start_ts = datetime.now(timezone.utc).isoformat()
    client.hset(key, "delete_version_start", start_ts)
    client.hset(key, "operation", "delete_version")
    _update_status(client, key, TaskStatus.OPERATION, 0.0)
    try:
        ds.delete_version(index)
        ts = datetime.now(timezone.utc).isoformat()
        client.hset(
            key,
            "delete_version",
            json.dumps({"index": index, "time": ts}),
        )
        client.hset(key, "delete_version_finish", ts)
        _update_status(client, key, TaskStatus.COMPLETED, 1.0)
        return {"versions": len(ds.versions)}
    except Exception as exc:
        _record_error(client, key, exc, ds)
        raise
    finally:
        if driver:
            driver.close()


@celery_app.task
def datasets_prune_versions_task(
    limit: int | None = None,
) -> dict:  # pragma: no cover - heavy
    """Prune stored versions for all datasets."""

    client = get_redis_client()
    if client is None:
        raise RuntimeError("Redis unavailable")

    names = [
        n.decode() if isinstance(n, bytes) else n for n in client.smembers("datasets")
    ]
    total_removed = 0
    details: dict[str, int] = {}

    for name in names:
        driver = get_neo4j_driver()
        ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
        ds.redis_client = client
        removed = ds.prune_versions(limit)
        details[name] = removed
        total_removed += removed
        if driver:
            driver.close()

    return {"datasets": len(names), "removed": total_removed, "details": details}


@celery_app.task
def datasets_prune_stale_task(days: int = 30) -> dict:  # pragma: no cover - heavy
    """Delete datasets not accessed within ``days`` days."""

    client = get_redis_client()
    if client is None:
        raise RuntimeError("Redis unavailable")

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    names = [
        n.decode() if isinstance(n, bytes) else n for n in client.smembers("datasets")
    ]
    removed: list[str] = []

    for name in names:
        try:
            ds = DatasetBuilder.from_redis(client, f"dataset:{name}", None)
        except Exception:
            continue
        if ds.accessed_at < cutoff:
            keys = [
                k.decode() if isinstance(k, bytes) else k
                for k in client.scan_iter(match=f"dataset:{name}*")
            ]
            pipe = client.pipeline()
            for k in keys:
                pipe.delete(k)
            pipe.srem("datasets", name)
            if ds.owner_id is not None:
                pipe.srem(f"user:{ds.owner_id}:datasets", name)
            pipe.execute()
            graph = get_redis_graph(name)
            if graph is not None:
                try:
                    graph.query(
                        "MATCH (n {dataset:$ds}) DETACH DELETE n",
                        {"ds": name},
                    )
                except Exception:
                    logger.exception("Failed to delete RedisGraph for %s", name)
            driver = get_neo4j_driver()
            if driver:
                try:
                    with driver.session() as session:
                        session.run("MATCH (n {dataset:$ds}) DETACH DELETE n", ds=name)
                finally:
                    driver.close()
            removed.append(name)

    return {"removed": removed, "count": len(removed)}


@celery_app.task
def dataset_extract_facts_task(  # pragma: no cover - heavy
    name: DatasetName,
    provider: str | None = None,
    profile: str | None = None,
    user_id: int | None = None,
) -> dict:
    """Run fact extraction asynchronously."""

    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    ds.redis_client = client
    llm_client = None
    if provider or profile:
        ds.configure_llm_service(provider=provider or "vllm", profile=profile)
        llm_client = ds.llm_service.client
    key = f"dataset:{name}:progress"
    _update_status(client, key, TaskStatus.EXTRACTING_FACTS, 0.0)
    try:
        ds.extract_facts(llm_client)
        _update_status(client, key, TaskStatus.EXTRACTING_FACTS, 0.5)
        ts = datetime.now(timezone.utc).isoformat()
        client.hset(
            key,
            "extract_facts",
            json.dumps({"time": ts, "done": True}),
        )
        _update_status(client, key, TaskStatus.COMPLETED, 1.0)
        return {"stage": ds.stage}
    except Exception as exc:
        _record_error(client, key, exc, ds)
        raise


@celery_app.task
def dataset_extract_entities_task(  # pragma: no cover - heavy
    name: DatasetName, model: str | None = None, user_id: int | None = None
) -> dict:
    """Run NER asynchronously."""

    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    ds.redis_client = client
    key = f"dataset:{name}:progress"
    _update_status(client, key, TaskStatus.EXTRACTING_ENTITIES, 0.0)
    try:
        ds.extract_entities(model=model)
        _update_status(client, key, TaskStatus.EXTRACTING_ENTITIES, 0.5)
        ts = datetime.now(timezone.utc).isoformat()
        client.hset(
            key,
            "extract_entities",
            json.dumps({"time": ts, "done": True}),
        )
        _update_status(client, key, TaskStatus.COMPLETED, 1.0)
        return {"stage": ds.stage}
    except Exception as exc:
        _record_error(client, key, exc, ds)
        raise


@celery_app.task
def dataset_delete_task(
    name: DatasetName, user_id: int | None = None
) -> dict:  # pragma: no cover - heavy
    """Remove a dataset from Redis and Neo4j."""

    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = None
    try:
        ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
    except KeyError:
        pass
    if user_id is not None and ds and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    prog_key = f"dataset:{name}:progress"
    start_ts = datetime.now(timezone.utc).isoformat()
    client.hset(prog_key, "delete_start", start_ts)
    _update_status(client, prog_key, TaskStatus.DELETING, 0.0)
    try:
        keys = []
        for key in client.scan_iter(match=f"dataset:{name}*"):
            k = key.decode() if isinstance(key, bytes) else key
            if k != prog_key:
                keys.append(k)
        pipe = client.pipeline()
        for k in keys:
            pipe.delete(k)
        pipe.srem("datasets", name)
        if ds and ds.owner_id is not None:
            pipe.srem(f"user:{ds.owner_id}:datasets", name)
        pipe.execute()
        graph = get_redis_graph(name)
        if graph is not None:
            try:
                graph.query("MATCH (n {dataset:$ds}) DETACH DELETE n", {"ds": name})
            except Exception:
                logger.exception("Failed to delete RedisGraph for %s", name)
        _update_status(client, prog_key, TaskStatus.DELETING, 0.5)

        driver = get_neo4j_driver()
        if driver:
            try:
                with driver.session() as session:
                    session.run(
                        "MATCH (n {dataset:$dataset}) DETACH DELETE n",
                        dataset=name,
                    )
            finally:
                driver.close()
        ts = datetime.now(timezone.utc).isoformat()
        client.hset(
            prog_key,
            "delete",
            json.dumps({"deleted": True, "time": ts}),
        )
        client.hset(prog_key, "delete_finish", ts)
        client.hset(
            f"graph:{name}:progress",
            "delete",
            json.dumps({"deleted": True, "time": ts}),
        )
        _update_status(client, prog_key, TaskStatus.COMPLETED, 1.0)

        return {"deleted": name}
    except Exception as exc:
        _record_error(client, prog_key, exc, ds)
        raise


@celery_app.task
def graph_save_neo4j_task(
    name: str, user_id: int | None = None
) -> dict:  # pragma: no cover - heavy
    """Persist a knowledge graph to Neo4j."""

    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = DatasetBuilder.from_redis(client, f"graph:{name}", driver)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    ds.redis_client = client
    if not driver:
        raise RuntimeError("Neo4j not configured")
    key = f"graph:{name}:progress"
    start_ts = datetime.now(timezone.utc).isoformat()
    client.hset(key, "save_neo4j_start", start_ts)
    _update_status(client, key, TaskStatus.SAVING_NEO4J, 0.0)
    try:
        neo4j_breaker.call(ds.save_neo4j, driver)
        _update_status(client, key, TaskStatus.SAVING_NEO4J, 0.5)
        driver.close()
        ds.redis_client = client
        ts = datetime.now(timezone.utc).isoformat()
        client.hset(
            key,
            "save_neo4j",
            json.dumps({"nodes": len(ds.graph.graph), "time": ts}),
        )
        client.hset(key, "save_neo4j_finish", ts)
        _update_status(client, key, TaskStatus.COMPLETED, 1.0)
        return {"nodes": len(ds.graph.graph)}
    except CircuitBreakerError as exc:
        _record_error(client, key, exc, ds)
        raise RuntimeError("neo4j circuit open") from exc
    except Exception as exc:
        _record_error(client, key, exc, ds)
        raise


@celery_app.task
def graph_load_neo4j_task(
    name: str, user_id: int | None = None
) -> dict:  # pragma: no cover - heavy
    """Load a knowledge graph from Neo4j."""

    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = DatasetBuilder.from_redis(client, f"graph:{name}", driver)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    ds.redis_client = client
    if not driver:
        raise RuntimeError("Neo4j not configured")
    key = f"graph:{name}:progress"
    start_ts = datetime.now(timezone.utc).isoformat()
    client.hset(key, "load_neo4j_start", start_ts)
    _update_status(client, key, TaskStatus.LOADING_NEO4J, 0.0)
    try:
        ds.load_neo4j(driver)
        _update_status(client, key, TaskStatus.LOADING_NEO4J, 0.5)
        driver.close()
        ds.neo4j_driver = None
        ds.redis_client = client
        ts = datetime.now(timezone.utc).isoformat()
        client.hset(
            key,
            "load_neo4j",
            json.dumps({"nodes": len(ds.graph.graph), "time": ts}),
        )
        client.hset(key, "load_neo4j_finish", ts)
        _update_status(client, key, TaskStatus.COMPLETED, 1.0)
        return {"nodes": len(ds.graph.graph)}
    except Exception as exc:
        _record_error(client, key, exc, ds)
        raise


@celery_app.task
def graph_delete_task(
    name: str, user_id: int | None = None
) -> dict:  # pragma: no cover - heavy
    """Remove a knowledge graph from Redis and Neo4j."""

    client = get_redis_client()
    driver = get_neo4j_driver()
    ds = DatasetBuilder.from_redis(client, f"graph:{name}", driver)
    if user_id is not None and ds.owner_id not in {None, user_id}:
        raise RuntimeError("Unauthorized")
    prog_key = f"graph:{name}:progress"
    start_ts = datetime.now(timezone.utc).isoformat()
    client.hset(prog_key, "delete_start", start_ts)
    _update_status(client, prog_key, TaskStatus.DELETING, 0.0)
    try:
        keys = list(client.scan_iter(match=f"graph:{name}*"))
        for key in keys:
            k = key.decode() if isinstance(key, bytes) else key
            if k != prog_key:
                client.delete(key)
        client.srem("graphs", name)
        if ds.owner_id is not None:
            client.srem(f"user:{ds.owner_id}:graphs", name)
        _update_status(client, prog_key, TaskStatus.DELETING, 0.5)

        driver = get_neo4j_driver()
        if driver:
            try:
                with driver.session() as session:
                    session.run(
                        "MATCH (n {dataset:$dataset}) DETACH DELETE n",
                        dataset=name,
                    )
            finally:
                driver.close()

        ts = datetime.now(timezone.utc).isoformat()
        client.hset(
            prog_key,
            "delete",
            json.dumps({"deleted": True, "time": ts}),
        )
        client.hset(prog_key, "delete_finish", ts)
        _update_status(client, prog_key, TaskStatus.COMPLETED, 1.0)

        return {"deleted": name}
    except Exception as exc:
        _record_error(client, prog_key, exc, ds)
        raise
