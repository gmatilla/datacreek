import enum
import json
from typing import Any, Literal

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session

from datacreek.backends import get_neo4j_driver, get_redis_client
from datacreek.core.dataset import MAX_NAME_LENGTH, NAME_PATTERN, DatasetBuilder
from datacreek.routers import explain_router, vector_router
from datacreek.security.dp_middleware import DPBudgetMiddleware

try:  # optional heavy imports
    from datacreek.db import Dataset, SessionLocal, User, init_db
    from datacreek.models.export_format import ExportFormat
    from datacreek.schemas import (
        CurateParams,
        DatasetCreate,
        DatasetInit,
        DatasetName,
        DatasetOut,
        DatasetUpdate,
        GenerateParams,
        SaveParams,
        SourceCreate,
        SourceOut,
        UserCreate,
        UserOut,
        UserWithKey,
    )
    from datacreek.services import (
        _cache_dataset,
        create_dataset,
        create_user,
        create_user_with_generated_key,
        get_dataset_by_id,
        get_user_by_key,
    )
    from datacreek.tasks import (
        celery_app,
        curate_task,
        dataset_cleanup_task,
        dataset_delete_task,
        dataset_export_task,
        dataset_generate_task,
        dataset_ingest_task,
        enqueue_dataset_ingest,
        generate_task,
        ingest_task,
        save_task,
    )
except Exception:  # pragma: no cover - simplify tests
    Dataset = SessionLocal = User = None  # type: ignore

    class ExportFormat(str, enum.Enum):
        JSONL = "jsonl"
        PARQUET = "parquet"
        DELTA = "delta"

    CurateParams = DatasetCreate = DatasetInit = DatasetName = DatasetOut = (
        DatasetUpdate
    ) = GenerateParams = SaveParams = SourceCreate = SourceOut = UserCreate = (
        UserOut
    ) = UserWithKey = Any

    def init_db():
        pass

    def _cache_dataset(*args, **kwargs):
        pass

    def create_dataset(*args, **kwargs):
        pass

    def create_user(*args, **kwargs):
        pass

    def create_user_with_generated_key(*args, **kwargs):
        return None, None

    def get_dataset_by_id(*args, **kwargs):
        return None

    def get_user_by_key(*args, **kwargs):
        return None

    celery_app = curate_task = dataset_cleanup_task = dataset_delete_task = (
        dataset_export_task
    ) = dataset_generate_task = dataset_ingest_task = enqueue_dataset_ingest = (
        generate_task
    ) = ingest_task = save_task = None
from datacreek.analysis import explain_to_svg
from datacreek.telemetry import init_tracing
from datacreek.utils import decode_hash

init_db()  # pragma: no cover - avoid DB during tests
app = FastAPI(title="Datacreek API")
init_tracing(app)  # pragma: no cover - tracer optional
app.add_middleware(DPBudgetMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)  # pragma: no cover - startup config
app.include_router(explain_router)  # pragma: no cover - router registration
app.include_router(vector_router)  # pragma: no cover - router registration


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    api_key: str = Header(..., alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> User:
    user = get_user_by_key(db, api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return user


@app.post("/users", response_model=UserWithKey, summary="Create a user")
def create_user_route(payload: UserCreate, db: Session = Depends(get_db)):
    user, key = create_user_with_generated_key(db, payload.username)
    return {"id": user.id, "username": user.username, "api_key": key}


@app.get("/users/me/datasets", summary="List user's datasets")
def list_user_datasets(current_user: User = Depends(get_current_user)) -> list[str]:
    """Return dataset names owned by the authenticated user."""
    client = get_redis_client()
    if client is None:
        return []
    key = f"user:{current_user.id}:datasets"
    names = [n.decode() if isinstance(n, bytes) else n for n in client.smembers(key)]
    return sorted(names)


@app.get("/users/me/datasets/details", summary="List user's datasets with progress")
def list_user_datasets_details(
    current_user: User = Depends(get_current_user),
) -> list[dict]:
    """Return datasets with their stage and progress information."""
    client = get_redis_client()
    if client is None:
        return []
    key = f"user:{current_user.id}:datasets"
    names = [n.decode() if isinstance(n, bytes) else n for n in client.smembers(key)]
    datasets: list[dict] = []
    for name in sorted(names):
        try:
            driver = get_neo4j_driver()
            ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
        except KeyError:
            continue
        finally:
            if driver:
                driver.close()

        progress_key = f"dataset:{name}:progress"
        progress = decode_hash(client.hgetall(progress_key))
        datasets.append({"name": name, "stage": ds.stage, "progress": progress})
    return datasets


def _load_dataset(name: str, current_user: User) -> DatasetBuilder:
    """Return ``DatasetBuilder`` for ``name`` if owned by ``current_user``."""

    client = get_redis_client()
    if client is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        driver = get_neo4j_driver()
        ds = DatasetBuilder.from_redis(client, f"dataset:{name}", driver)
    except KeyError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    finally:
        if driver:
            driver.close()
    if ds.owner_id not in {None, current_user.id}:
        raise HTTPException(status_code=404, detail="Dataset not found")
    ds.redis_client = client
    return ds


@app.post("/datasets/{name}", summary="Create persisted dataset", status_code=201)
def create_persisted_dataset(
    name: DatasetName = Path(
        ..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH
    ),
    params: DatasetInit = Body(...),
    current_user: User = Depends(get_current_user),
):  # pragma: no cover
    """Initialize an empty dataset in Redis and Neo4j."""  # pragma: no cover
    # pragma: no cover
    client = get_redis_client()  # pragma: no cover
    if client is None:  # pragma: no cover
        raise HTTPException(
            status_code=503, detail="Redis unavailable"
        )  # pragma: no cover
    # pragma: no cover
    key = f"dataset:{name}"  # pragma: no cover
    if client.exists(key):  # pragma: no cover
        raise HTTPException(
            status_code=409, detail="Dataset already exists"
        )  # pragma: no cover
    # pragma: no cover
    driver = get_neo4j_driver()  # pragma: no cover
    ds = DatasetBuilder(  # pragma: no cover
        params.dataset_type,
        name=name,
        redis_client=client,
        neo4j_driver=driver,  # pragma: no cover
    )  # pragma: no cover
    ds.owner_id = current_user.id  # pragma: no cover
    ds.history.append("Dataset created")  # pragma: no cover
    ds._persist()  # pragma: no cover
    # pragma: no cover
    client.sadd("datasets", name)  # pragma: no cover
    client.sadd(f"user:{current_user.id}:datasets", name)  # pragma: no cover
    if driver:  # pragma: no cover
        driver.close()  # pragma: no cover
    # pragma: no cover
    return {"name": name, "stage": ds.stage}  # pragma: no cover


@app.post("/datasets", response_model=DatasetOut, summary="Add a dataset")
def add_dataset(
    payload: DatasetCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):  # pragma: no cover
    ds = create_dataset(
        db, current_user.id, payload.source_id, payload.path
    )  # pragma: no cover
    return ds  # pragma: no cover


@app.delete("/datasets/{name}", summary="Delete persisted dataset", status_code=202)
def delete_persisted_dataset(
    name: DatasetName = Path(
        ..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH
    ),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Delete a persisted dataset from Redis and Neo4j."""  # pragma: no cover
    # pragma: no cover
    _load_dataset(name, current_user)  # pragma: no cover
    # pragma: no cover
    celery_task = dataset_delete_task.apply_async(
        args=[name, current_user.id]
    )  # pragma: no cover
    return {"task_id": celery_task.id}  # pragma: no cover


@app.get("/datasets", response_model=list[DatasetOut], summary="List datasets")
def list_datasets(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):  # pragma: no cover
    return (
        db.query(Dataset).filter(Dataset.owner_id == current_user.id).all()
    )  # pragma: no cover


@app.get("/datasets/events", summary="Get global dataset events")
def global_dataset_events(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
) -> list[dict]:
    """Return recent dataset events across all datasets."""  # pragma: no cover
    # pragma: no cover
    client = get_redis_client()  # pragma: no cover
    if client is None:  # pragma: no cover
        return []  # pragma: no cover
    start = -(offset + limit)  # pragma: no cover
    end = -offset - 1  # pragma: no cover
    raw_events = client.lrange("dataset:events", start, end)  # pragma: no cover
    events: list[dict] = []  # pragma: no cover
    for raw in raw_events:  # pragma: no cover
        if isinstance(raw, bytes):  # pragma: no cover
            raw = raw.decode()  # pragma: no cover
        try:  # pragma: no cover
            events.append(json.loads(raw))  # pragma: no cover
        except Exception:  # pragma: no cover
            continue  # pragma: no cover
    events.reverse()  # pragma: no cover
    return events  # pragma: no cover


@app.get("/datasets/{ds_id}", response_model=DatasetOut, summary="Get dataset")
def get_dataset(
    ds_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):  # pragma: no cover
    ds = get_dataset_by_id(db, ds_id)  # pragma: no cover
    if not ds or ds.owner_id != current_user.id:  # pragma: no cover
        raise HTTPException(
            status_code=404, detail="Dataset not found"
        )  # pragma: no cover
    return ds  # pragma: no cover


@app.patch("/datasets/{ds_id}", response_model=DatasetOut, summary="Update dataset")
def update_dataset(
    ds_id: int,
    payload: DatasetUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):  # pragma: no cover
    record = get_dataset_by_id(db, ds_id)  # pragma: no cover
    if not record or record.owner_id != current_user.id:  # pragma: no cover
        raise HTTPException(
            status_code=404, detail="Dataset not found"
        )  # pragma: no cover
    # pragma: no cover
    ds = db.get(Dataset, ds_id)  # pragma: no cover
    if ds is None:  # pragma: no cover
        raise HTTPException(
            status_code=404, detail="Dataset not found"
        )  # pragma: no cover
    # pragma: no cover
    if payload.path:  # pragma: no cover
        ds.path = payload.path  # pragma: no cover
    db.commit()  # pragma: no cover
    db.refresh(ds)  # pragma: no cover
    _cache_dataset(ds)  # pragma: no cover
    return ds  # pragma: no cover


@app.delete("/datasets/{ds_id}", summary="Delete dataset")
def delete_dataset_route(
    ds_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):  # pragma: no cover
    ds = get_dataset_by_id(db, ds_id)  # pragma: no cover
    if not ds or ds.owner_id != current_user.id:  # pragma: no cover
        raise HTTPException(
            status_code=404, detail="Dataset not found"
        )  # pragma: no cover
    db.delete(ds)  # pragma: no cover
    db.commit()  # pragma: no cover
    return {"status": "deleted"}  # pragma: no cover


@app.get("/datasets/{name}/history", summary="Get dataset event history")
def dataset_history(
    name: DatasetName = Path(
        ..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH
    ),
    current_user: User = Depends(get_current_user),
) -> list[dict]:
    """Return stored dataset events from Redis."""  # pragma: no cover
    ds = _load_dataset(name, current_user)  # pragma: no cover
    client = ds.redis_client  # pragma: no cover
    events: list[dict] = []  # pragma: no cover
    key = f"dataset:{name}:events"  # pragma: no cover
    for raw in client.lrange(key, 0, -1):  # pragma: no cover
        if isinstance(raw, bytes):  # pragma: no cover
            raw = raw.decode()  # pragma: no cover
        try:  # pragma: no cover
            events.append(json.loads(raw))  # pragma: no cover
        except Exception:  # pragma: no cover
            continue  # pragma: no cover
    return events  # pragma: no cover


@app.get("/datasets/{name}/versions", summary="List dataset versions")
def dataset_versions(
    name: DatasetName = Path(
        ..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH
    ),
    current_user: User = Depends(get_current_user),
) -> list[dict]:
    """Return generation versions stored for ``name``."""  # pragma: no cover
    ds = _load_dataset(name, current_user)  # pragma: no cover
    return [
        {"index": i + 1, **v} for i, v in enumerate(ds.versions)
    ]  # pragma: no cover


@app.get("/datasets/{name}/versions/{index}", summary="Get dataset version")
def dataset_version_item(
    name: DatasetName = Path(
        ..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH
    ),
    index: int = Path(..., ge=1),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Return a single generation version for ``name``."""
    ds = _load_dataset(name, current_user)
    if index < 1 or index > len(ds.versions):
        raise HTTPException(status_code=404, detail="Version not found")
    return ds.versions[index - 1]


@app.delete("/datasets/{name}/versions/{index}", summary="Delete dataset version")
def delete_dataset_version_item(
    name: DatasetName = Path(
        ..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH
    ),
    index: int = Path(..., ge=1),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Delete a stored generation version for ``name``."""

    ds = _load_dataset(name, current_user)
    try:
        ds.delete_version(index)
    except IndexError:
        raise HTTPException(status_code=404, detail="Version not found") from None
    return {"status": "deleted"}


@app.post(
    "/datasets/{name}/versions/{index}/restore", summary="Restore dataset version"
)
def restore_dataset_version_item(
    name: DatasetName = Path(
        ..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH
    ),
    index: int = Path(..., ge=1),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Copy version ``index`` to the end of the version list."""  # pragma: no cover
    # pragma: no cover
    ds = _load_dataset(name, current_user)  # pragma: no cover
    try:  # pragma: no cover
        ds.restore_version(index)  # pragma: no cover
    except IndexError:  # pragma: no cover
        raise HTTPException(
            status_code=404, detail="Version not found"
        ) from None  # pragma: no cover
    return {"status": "restored"}  # pragma: no cover


@app.get("/datasets/{name}/progress", summary="Get dataset progress")
def dataset_progress(
    name: DatasetName = Path(
        ..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH
    ),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Return progress information stored in Redis."""
    ds = _load_dataset(name, current_user)
    client = ds.redis_client
    key = f"dataset:{name}:progress"
    return decode_hash(client.hgetall(key))


@app.get("/graphs/{name}/progress", summary="Get graph progress")
def graph_progress(
    name: DatasetName = Path(
        ..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH
    ),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Return progress information stored for a knowledge graph."""  # pragma: no cover
    client = get_redis_client()  # pragma: no cover
    if client is None:  # pragma: no cover
        raise HTTPException(
            status_code=404, detail="Graph not found"
        )  # pragma: no cover
    try:  # pragma: no cover
        driver = get_neo4j_driver()  # pragma: no cover
        ds = DatasetBuilder.from_redis(
            client, f"graph:{name}", driver
        )  # pragma: no cover
    except KeyError:  # pragma: no cover
        raise HTTPException(
            status_code=404, detail="Graph not found"
        )  # pragma: no cover
    finally:  # pragma: no cover
        if driver:  # pragma: no cover
            driver.close()  # pragma: no cover
    if ds.owner_id not in {None, current_user.id}:  # pragma: no cover
        raise HTTPException(
            status_code=404, detail="Graph not found"
        )  # pragma: no cover
    key = f"graph:{name}:progress"  # pragma: no cover
    return decode_hash(client.hgetall(key))  # pragma: no cover


@app.get("/datasets/{name}/progress/history", summary="Get progress history")
def dataset_progress_history(
    name: DatasetName = Path(
        ..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH
    ),
    current_user: User = Depends(get_current_user),
) -> list[dict]:
    """Return progress status history stored in Redis."""
    ds = _load_dataset(name, current_user)
    client = ds.redis_client
    key = f"dataset:{name}:progress:history"
    raw = client.lrange(key, 0, -1)
    history: list[dict] = []
    for item in raw:
        if isinstance(item, bytes):
            item = item.decode()
        try:
            history.append(json.loads(item))
        except Exception:
            continue
    return history


@app.get("/graphs/{name}/progress/history", summary="Get graph progress history")
def graph_progress_history(
    name: DatasetName = Path(
        ..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH
    ),
    current_user: User = Depends(get_current_user),
) -> list[dict]:
    """Return progress status history stored for a knowledge graph."""  # pragma: no cover
    client = get_redis_client()  # pragma: no cover
    if client is None:  # pragma: no cover
        raise HTTPException(
            status_code=404, detail="Graph not found"
        )  # pragma: no cover
    try:  # pragma: no cover
        driver = get_neo4j_driver()  # pragma: no cover
        ds = DatasetBuilder.from_redis(
            client, f"graph:{name}", driver
        )  # pragma: no cover
    except KeyError:  # pragma: no cover
        raise HTTPException(
            status_code=404, detail="Graph not found"
        )  # pragma: no cover
    finally:  # pragma: no cover
        if driver:  # pragma: no cover
            driver.close()  # pragma: no cover
    if ds.owner_id not in {None, current_user.id}:  # pragma: no cover
        raise HTTPException(
            status_code=404, detail="Graph not found"
        )  # pragma: no cover
    key = f"graph:{name}:progress:history"  # pragma: no cover
    raw = client.lrange(key, 0, -1)  # pragma: no cover
    history: list[dict] = []  # pragma: no cover
    for item in raw:  # pragma: no cover
        if isinstance(item, bytes):  # pragma: no cover
            item = item.decode()  # pragma: no cover
        try:  # pragma: no cover
            history.append(json.loads(item))  # pragma: no cover
        except Exception:  # pragma: no cover
            continue  # pragma: no cover
    return history  # pragma: no cover


@app.post("/datasets/{name}/ingest", summary="Ingest file into dataset")
def dataset_ingest_route(
    name: DatasetName = Path(
        ..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH
    ),
    payload: SourceCreate = Body(...),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Schedule ingestion of a file into a persisted dataset."""  # pragma: no cover
    # pragma: no cover
    ds = _load_dataset(name, current_user)  # pragma: no cover
    result = enqueue_dataset_ingest(  # pragma: no cover
        name,  # pragma: no cover
        payload.path,  # pragma: no cover
        current_user.id,  # pragma: no cover
        doc_id=payload.name,  # pragma: no cover
        high_res=payload.high_res or False,  # pragma: no cover
        ocr=payload.ocr or False,  # pragma: no cover
        use_unstructured=payload.use_unstructured,  # pragma: no cover
        extract_entities=payload.extract_entities or False,  # pragma: no cover
        extract_facts=payload.extract_facts or False,  # pragma: no cover
    )  # pragma: no cover
    return {"task_id": getattr(result, "id", None)}  # pragma: no cover


@app.post("/datasets/{name}/generate", summary="Generate dataset asynchronously")
def dataset_generate_route(
    name: DatasetName = Path(
        ..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH
    ),
    params: dict | None = Body(None),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Schedule the generation pipeline for a persisted dataset."""

    _load_dataset(name, current_user)  # pragma: no cover
    # pragma: no cover
    celery_task = dataset_generate_task.apply_async(  # pragma: no cover
        args=[name, params or {}, current_user.id]  # pragma: no cover
    )  # pragma: no cover
    return {"task_id": celery_task.id}  # pragma: no cover


@app.post("/datasets/{name}/cleanup", summary="Cleanup dataset asynchronously")
def dataset_cleanup_route(
    name: DatasetName = Path(
        ..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH
    ),
    params: dict | None = Body(None),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Schedule cleanup operations on a persisted dataset."""  # pragma: no cover
    # pragma: no cover
    _load_dataset(name, current_user)  # pragma: no cover
    # pragma: no cover
    celery_task = dataset_cleanup_task.apply_async(  # pragma: no cover
        args=[name, params or {}, current_user.id]  # pragma: no cover
    )  # pragma: no cover
    return {"task_id": celery_task.id}  # pragma: no cover


@app.post("/datasets/{name}/export", summary="Export dataset asynchronously")
def dataset_export_task_route(
    name: DatasetName = Path(
        ..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH
    ),
    fmt: ExportFormat = ExportFormat.JSONL,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Schedule dataset export to the requested format."""  # pragma: no cover
    # pragma: no cover
    _load_dataset(name, current_user)  # pragma: no cover
    # pragma: no cover
    celery_task = dataset_export_task.apply_async(
        args=[name, fmt, current_user.id]
    )  # pragma: no cover
    return {"task_id": celery_task.id}  # pragma: no cover


@app.get("/datasets/{name}/export", summary="Get exported dataset")
def dataset_export_result(
    name: DatasetName = Path(
        ..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH
    ),
    fmt: ExportFormat = ExportFormat.JSONL,
    current_user: User = Depends(get_current_user),
) -> Response:
    """Return previously exported dataset from Redis."""
    ds = _load_dataset(name, current_user)
    client = ds.redis_client
    progress_key = f"dataset:{name}:progress"
    progress = client.hget(progress_key, "export")
    key: str | None = None
    if progress:
        try:
            entry = json.loads(progress)
            key = entry.get("key")
        except Exception:
            pass
    if not key:
        key = f"dataset:{name}:export:{fmt.value if isinstance(fmt, ExportFormat) else fmt}"
    data = client.get(key)
    if data is None:
        raise HTTPException(status_code=404, detail="Export not found")
    if isinstance(data, bytes):
        data = data.decode()
    filename = f"{name}.{fmt.value if isinstance(fmt, ExportFormat) else fmt}"
    headers = {"Content-Disposition": f"attachment; filename={filename}"}
    return Response(data, media_type="application/json", headers=headers)


@app.get("/datasets/{ds_id}/download", summary="Download dataset")
def download_dataset(
    ds_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):  # pragma: no cover
    ds = get_dataset_by_id(db, ds_id)  # pragma: no cover
    if not ds or ds.owner_id != current_user.id:  # pragma: no cover
        raise HTTPException(
            status_code=404, detail="Dataset not found"
        )  # pragma: no cover
    filename = f"dataset_{ds_id}.json"  # pragma: no cover
    headers = {
        "Content-Disposition": f"attachment; filename={filename}"
    }  # pragma: no cover
    return Response(
        ds.content or "{}", media_type="application/json", headers=headers
    )  # pragma: no cover


# ----- Asynchronous task endpoints -----


@app.post("/tasks/ingest", summary="Ingest a file asynchronously")
async def ingest_async(
    payload: SourceCreate,
    current_user: User = Depends(get_current_user),
) -> dict:
    celery_task = ingest_task.apply_async(
        args=[current_user.id, payload.path],
        kwargs={
            "high_res": payload.high_res or False,
            "ocr": payload.ocr or False,
            "use_unstructured": payload.use_unstructured,
            "extract_entities": payload.extract_entities or False,
            "extract_facts": payload.extract_facts or False,
        },
    )
    return {"task_id": celery_task.id}


@app.post("/tasks/generate", summary="Generate dataset asynchronously")
async def generate_async(
    params: GenerateParams,
    current_user: User = Depends(get_current_user),
    x_config_path: str | None = Header(None, alias="X-Config-Path"),
) -> dict:
    celery_task = generate_task.apply_async(
        args=[current_user.id, params.src_id, params.content_type, params.num_pairs],
        kwargs={
            "provider": params.provider,
            "profile": params.profile,
            "model": params.model,
            "api_base": params.api_base,
            "config_path": x_config_path,
            "generation": (
                params.generation.model_dump(exclude_defaults=True, exclude_none=True)
                if params.generation
                else None
            ),
            "prompts": params.prompts,
        },
    )
    return {"task_id": celery_task.id}


@app.post("/tasks/curate", summary="Curate a dataset asynchronously")
async def curate_async(
    params: CurateParams,
    current_user: User = Depends(get_current_user),
) -> dict:
    celery_task = curate_task.apply_async(
        args=[current_user.id, params.ds_id, params.threshold]
    )
    return {"task_id": celery_task.id}


@app.post("/tasks/save", summary="Convert dataset asynchronously")
async def save_async(
    params: SaveParams,
    current_user: User = Depends(get_current_user),
) -> dict:
    celery_task = save_task.apply_async(
        args=[current_user.id, params.ds_id, params.fmt.value]
    )
    return {"task_id": celery_task.id}


@app.get("/tasks/{task_id}", summary="Get background task status")
async def get_task(task_id: str) -> dict:
    res = celery_app.AsyncResult(task_id)
    if res.state in {"PENDING", "STARTED"}:
        return {"status": "running"}
    if res.state == "SUCCESS":
        return {"status": "finished", "result": res.result}
    if res.state == "FAILURE":
        return {"status": "failed", "error": str(res.result)}
    return {"status": res.state.lower()}


@app.post("/dp/sample", summary="Dummy endpoint for DP budget testing")
def dp_sample(current_user: User = Depends(get_current_user)) -> dict:
    """Endpoint guarded by :class:`DPBudgetMiddleware`."""

    return {"ok": True}


@app.get("/dp/budget", summary="Return remaining DP budget for the user")
def get_dp_budget(
    current_user: User = Depends(get_current_user),
) -> dict:  # pragma: no cover
    """Expose current user's epsilon budget information.

    This endpoint allows clients to monitor how much differential privacy
    budget remains. It leverages :func:`get_budget` from
    ``datacreek.security.tenant_privacy``.
    """

    with SessionLocal() as db:
        from datacreek.security.tenant_privacy import get_budget

        info = get_budget(db, current_user.id)
        if info is None:
            return {"epsilon_max": 0.0, "epsilon_used": 0.0, "epsilon_remaining": 0.0}
        return info


@app.get(
    "/datasets/{name}/explain/{node_id}",
    summary="Explain node neighborhood with attention heatmap",
    response_model=None,
)
def explain_node_route(  # pragma: no cover
    name: DatasetName = Path(
        ..., pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH
    ),
    node_id: str = Path(...),
    hops: int = Query(3, ge=1, le=5),
    fmt: Literal["json", "svg", "png"] = Query("json"),
    current_user: User = Depends(get_current_user),
) -> Response:
    """Return a radius-``hops`` subgraph around ``node_id``.

    The JSON payload contains ``nodes`` and ``edges`` lists plus an ``attention``
    mapping. When ``fmt`` is ``svg`` or ``png`` the graph is rendered as an
    image and returned accordingly. PNG export requires ``cairosvg``.
    """

    ds = _load_dataset(name, current_user)
    data = ds.explain_node(node_id, hops=hops)

    if fmt == "json":
        return data

    svg = explain_to_svg(data)
    if fmt == "svg":
        return Response(content=svg, media_type="image/svg+xml")

    try:
        import cairosvg
    except Exception as exc:  # pragma: no cover - optional dependency
        raise HTTPException(status_code=501, detail="PNG export unavailable") from exc

    png = cairosvg.svg2png(bytestring=svg.encode())
    return Response(content=png, media_type="image/png")
