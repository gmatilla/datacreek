from typing import Annotated, Literal

try:  # optional dependency
    from pydantic import BaseModel, ConfigDict, StringConstraints, confloat, constr
except Exception:  # pragma: no cover - optional dependency missing

    class BaseModel:  # lightweight stub for missing pydantic
        pass

    class StringConstraints:  # type: ignore
        def __init__(self, **_kwargs):
            pass

    def confloat(*_args, **_kwargs):  # type: ignore
        return float

    def constr(*_args, **_kwargs):  # type: ignore
        return str

    class ConfigDict(dict):  # type: ignore
        pass


from datacreek.core.dataset import MAX_NAME_LENGTH, NAME_PATTERN

# Dataset identifier constraints reused by API paths and tasks
DatasetName = Annotated[
    str,
    StringConstraints(pattern=NAME_PATTERN.pattern, max_length=MAX_NAME_LENGTH),
]

from datacreek.config_models import GenerationSettingsModel
from datacreek.models.export_format import ExportFormat
from datacreek.pipelines import DatasetType


class UserCreate(BaseModel):
    username: constr(min_length=1)


class UserOut(BaseModel):
    id: int
    username: str

    model_config = ConfigDict(from_attributes=True)


class SourceCreate(BaseModel):
    path: constr(min_length=1)
    name: str | None = None
    high_res: bool | None = False
    ocr: bool | None = False
    use_unstructured: bool | None = None
    extract_entities: bool | None = False
    extract_facts: bool | None = False


class SourceOut(BaseModel):
    id: int

    model_config = ConfigDict(from_attributes=True)


from datacreek.models.content_type import ContentType


class GenerateParams(BaseModel):
    src_id: int
    content_type: ContentType = ContentType.QA
    num_pairs: int | None = None
    provider: constr(min_length=1) | None = None
    profile: constr(min_length=1) | None = None
    model: constr(min_length=1) | None = None
    api_base: constr(min_length=1) | None = None
    generation: GenerationSettingsModel | None = None
    prompts: dict | None = None


class DatasetOut(BaseModel):
    id: int
    path: str

    model_config = ConfigDict(from_attributes=True)


class CurateParams(BaseModel):
    ds_id: int
    threshold: confloat(ge=0, le=1) | None = None


class SaveParams(BaseModel):
    ds_id: int
    fmt: ExportFormat = ExportFormat.JSONL


class UserWithKey(UserOut):
    api_key: str


class DatasetCreate(BaseModel):
    source_id: int
    path: constr(min_length=1)


class DatasetUpdate(BaseModel):
    path: constr(min_length=1) | None = None


class DatasetInit(BaseModel):
    """Parameters for creating a persisted dataset."""

    dataset_type: DatasetType = DatasetType.TEXT
