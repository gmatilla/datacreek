from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import traceback
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

try:
    import redis
except Exception:  # pragma: no cover - optional dependency missing

    class _RedisStub:
        Redis = object

    redis = _RedisStub()  # type: ignore

try:
    from pydantic import BaseModel, ConfigDict, field_validator
except Exception:  # pragma: no cover - optional dependency missing

    class BaseModel:
        pass

    ConfigDict = dict

    def field_validator(*args, **kwargs):  # type: ignore[unused-argument]
        def decorator(fn):
            return fn

        return decorator


if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from datacreek.core.dataset import DatasetBuilder

from datacreek.backends import get_redis_client as backend_get_redis_client
from datacreek.core.cleanup import cleanup_knowledge_graph
from datacreek.core.create import process_file, process_file_async
from datacreek.core.curate import curate_qa_pairs, curate_qa_pairs_async
from datacreek.core.knowledge_graph import KnowledgeGraph, verify_thresholds
from datacreek.core.save_as import convert_format
from datacreek.models.content_type import ContentType
from datacreek.models.qa import QAPair
from datacreek.models.results import (
    ConversationResult,
    COTGenerationResult,
    CurationMetrics,
    CurationResult,
    PrefListResult,
    PrefPairResult,
    QAGenerationResult,
)
from datacreek.utils import create_progress
from datacreek.utils.config import (
    get_format_settings,
    get_redis_config,
    load_config_with_overrides,
)

logger = logging.getLogger(__name__)


def get_redis_client() -> redis.Redis | None:
    """Return a Redis client from backend configuration."""
    try:
        client = backend_get_redis_client()
        client.ping()
        return client
    except Exception:
        return None


class TrainingGoal(str, Enum):
    """Supported training objectives."""

    SFT = "sft"
    CPT = "cpt"
    RLAIF = "rlaif"
    GRPO = "grpo"
    PPO = "ppo"
    DPO = "dpo"
    ORPO = "orpo"
    DPO_SFT = "dpo_sft"
    RRHF = "rrhf"


class DatasetType(str, Enum):
    """Type of dataset produced by generation pipelines."""

    QA = "qa"
    COT = "cot"
    VQA = "vqa"
    TEXT = "text"
    KG = "kg"
    PREF_PAIR = "pref_pair"
    PREF_LIST = "pref_list"
    TOOL = "tool"
    CONVERSATION = "conversation"
    MULTI_TOOL = "multi_tool"


class PipelineStep(str, Enum):
    """Valid generation pipeline steps."""

    INGEST = "ingest"
    TO_KG = "to_kg"
    KG_CLEANUP = "kg_cleanup"
    VERIFY_THRESHOLDS = "verify_thresholds"
    GENERATE_QA = "generate_qa"
    GENERATE_COT = "generate_cot"
    GENERATE_VQA = "generate_vqa"
    GENERATE_FROM_KG = "generate_from_kg"
    GENERATE_CANDIDATES = "generate_candidates"
    LABEL_PAIRS = "label_pairs"
    RANK_RESPONSES = "rank_responses"
    GENERATE_TOOL_CALL = "generate_tool_call"
    GENERATE_CONVERSATION = "generate_conversation"
    GENERATE_MULTI_TOOL = "generate_multi_tool"
    CURATE = "curate"
    SAVE = "save"


@dataclass
class GenerationPipeline:
    """Definition of a generation pipeline."""

    dataset_type: DatasetType
    steps: List[PipelineStep]
    compatible_trainings: List[TrainingGoal]
    description: str


@dataclass
class ProcessOptions:
    """Grouped parameters for ``process_file`` calls."""

    config_path: Path | None = None
    provider: str | None = None
    profile: str | None = None
    api_base: str | None = None
    model: str | None = None
    num_pairs: int | None = None
    overrides: Dict[str, Any] | None = None
    verbose: bool = False
    async_mode: bool = False
    kg: KnowledgeGraph | None = None
    multi_answer: bool = False
    batch_size: int | None = None
    inference_batch: int | None = None
    keep_ratings: bool = False
    dedup_similarity: float = 1.0
    curation_temperature: float | None = None
    curation_threshold: float | None = None
    resume_curation: bool = False


@dataclass
class GenerationOptions:
    """Validated parameters for ``DatasetBuilder.run_post_kg_pipeline``."""

    config_path: Path | None = None
    provider: str | None = None
    profile: str | None = None
    api_base: str | None = None
    model: str | None = None
    num_pairs: int | None = None
    threshold: float | None = None
    fmt: str | None = None
    overrides: Dict[str, Any] | None = None
    verbose: bool = False
    async_mode: bool = False
    multi_answer: bool = False
    batch_size: int | None = None
    inference_batch: int | None = None
    start_step: PipelineStep | None = None
    pipeline_config_path: Path | None = None
    dedup_similarity: float = 1.0
    keep_ratings: bool = False


class GenerationOptionsModel(BaseModel):
    """Pydantic model for :class:`GenerationOptions`."""

    model_config = ConfigDict(extra="ignore")

    config_path: str | None = None
    provider: str | None = None
    profile: str | None = None
    api_base: str | None = None
    model: str | None = None
    num_pairs: int | None = None
    threshold: float | None = None
    fmt: str | None = None
    overrides: Dict[str, Any] | None = None
    verbose: bool = False
    async_mode: bool = False
    multi_answer: bool = False
    batch_size: int | None = None
    inference_batch: int | None = None
    start_step: PipelineStep | None = None
    pipeline_config_path: str | None = None
    dedup_similarity: float = 1.0
    keep_ratings: bool = False

    @field_validator("start_step", mode="before")
    def _parse_step(cls, v: Any) -> PipelineStep | None:
        if v is None:
            return None
        if isinstance(v, PipelineStep):
            return v
        try:
            return PipelineStep(v.lower())
        except Exception:
            raise ValueError(f"Invalid pipeline step: {v}") from None

    def to_options(self) -> GenerationOptions:
        data = self.model_dump()
        if data.get("config_path"):
            data["config_path"] = Path(data["config_path"])
        if data.get("pipeline_config_path"):
            data["pipeline_config_path"] = Path(data["pipeline_config_path"])
        return GenerationOptions(**data)


# Expected dataclass types for generation results
STEP_DATACLASSES = {
    PipelineStep.GENERATE_QA: QAGenerationResult,
    PipelineStep.GENERATE_FROM_KG: QAGenerationResult,
    PipelineStep.GENERATE_COT: COTGenerationResult,
    PipelineStep.GENERATE_TOOL_CALL: ConversationResult,
    PipelineStep.GENERATE_CONVERSATION: ConversationResult,
    PipelineStep.GENERATE_MULTI_TOOL: ConversationResult,
}

# Expected top-level fields for validation
STEP_FIELDS = {
    PipelineStep.GENERATE_QA: {"qa_pairs"},
    PipelineStep.GENERATE_FROM_KG: {"qa_pairs"},
    PipelineStep.GENERATE_COT: {"cot_examples"},
    PipelineStep.GENERATE_TOOL_CALL: {"conversations"},
    PipelineStep.GENERATE_CONVERSATION: {"conversations"},
    PipelineStep.GENERATE_MULTI_TOOL: {"conversations"},
    PipelineStep.GENERATE_CANDIDATES: {"pairs", "responses"},
}


def _validate_step_result(
    dataset_type: DatasetType, step: PipelineStep, result: Any
) -> Any:
    """Validate ``result`` for ``step`` and return it unchanged.

    Dataclass instances are accepted and preserved so later steps can rely on
    structured types. If ``result`` is a plain mapping, it is checked directly.
    """

    step_name = step.value
    expected_cls = STEP_DATACLASSES.get(step)
    if step is PipelineStep.GENERATE_CANDIDATES:
        expected_cls = (
            PrefPairResult if dataset_type == DatasetType.PREF_PAIR else PrefListResult
        )

    is_dc = is_dataclass(result)
    if is_dc:
        if expected_cls and not isinstance(result, expected_cls):
            raise ValueError(
                f"{step_name}: expected {expected_cls.__name__}, got {type(result).__name__}"
            )
        result_dict = asdict(result)
    else:
        result_dict = result

    expected_fields = STEP_FIELDS.get(step)
    if expected_fields:
        if not isinstance(result_dict, dict):
            raise ValueError(
                f"{step_name}: expected mapping, got {type(result).__name__}"
            )
        if step is PipelineStep.GENERATE_CANDIDATES:
            if not ("pairs" in result_dict or "responses" in result_dict):
                raise ValueError(
                    f"{step_name}: expected candidate pairs, got {type(result).__name__}"
                )
        else:
            missing = [f for f in expected_fields if f not in result_dict]
            if missing:
                raise ValueError(f"{step_name}: missing fields {', '.join(missing)}")
        for f in expected_fields:
            if f in result_dict:
                if not isinstance(result_dict[f], list):
                    raise ValueError(f"{step_name}: field '{f}' must be a list")
                if f == "qa_pairs":
                    for item in result_dict[f]:
                        if is_dataclass(item):
                            item_dict = asdict(item)
                        elif isinstance(item, dict):
                            item_dict = item
                        else:
                            raise ValueError(
                                f"{step_name}: qa_pairs items must be mappings"
                            )
                        if (
                            "question" not in item_dict
                            or "answer" not in item_dict
                            or not str(item_dict["question"]).strip()
                            or not str(item_dict["answer"]).strip()
                        ):
                            raise ValueError(f"{step_name}: invalid QA pair item")
                        if (
                            len(str(item_dict["question"])) > 1000
                            or len(str(item_dict["answer"])) > 5000
                        ):
                            raise ValueError(f"{step_name}: QA pair too long")
                if f == "cot_examples":
                    for item in result_dict[f]:
                        if is_dataclass(item):
                            item_dict = asdict(item)
                        elif isinstance(item, dict):
                            item_dict = item
                        else:
                            raise ValueError(
                                f"{step_name}: cot_examples items must be mappings"
                            )
                        required = {"question", "reasoning", "answer"}
                        if (
                            any(k not in item_dict for k in required)
                            or not str(item_dict["question"]).strip()
                            or not str(item_dict["answer"]).strip()
                            or not str(item_dict["reasoning"]).strip()
                        ):
                            raise ValueError(f"{step_name}: invalid COT example item")

        if step is PipelineStep.GENERATE_CANDIDATES:
            if dataset_type == DatasetType.PREF_PAIR:
                pairs = result_dict.get("pairs", [])
                if not isinstance(pairs, list):
                    raise ValueError(f"{step_name}: field 'pairs' must be a list")
                for item in pairs:
                    if not isinstance(item, dict):
                        raise ValueError(f"{step_name}: pairs items must be mappings")
                    if (
                        "question" not in item
                        or "chosen" not in item
                        or "rejected" not in item
                        or not str(item["question"]).strip()
                        or not str(item["chosen"]).strip()
                        or not str(item["rejected"]).strip()
                    ):
                        raise ValueError(f"{step_name}: invalid pair item")
            else:
                responses = result_dict.get("responses", [])
                if not isinstance(responses, list):
                    raise ValueError(f"{step_name}: field 'responses' must be a list")
                for resp in responses:
                    if not isinstance(resp, dict):
                        raise ValueError(
                            f"{step_name}: responses items must be mappings"
                        )
                    if not str(resp.get("question", "")).strip():
                        raise ValueError(f"{step_name}: response question is empty")
                    answers = resp.get("answers")
                    if not isinstance(answers, list) or not answers:
                        raise ValueError(
                            f"{step_name}: response answers must be a non-empty list"
                        )
                    for ans in answers:
                        if not isinstance(ans, dict):
                            raise ValueError(
                                f"{step_name}: answers items must be mappings"
                            )
                        if not str(ans.get("text", "")).strip():
                            raise ValueError(f"{step_name}: answer text is empty")

    return result


def _serialize(data: Any) -> Any:
    """Recursively convert dataclasses to dictionaries for JSON dumping."""
    if is_dataclass(data):
        return asdict(data)
    if isinstance(data, dict):
        return {k: _serialize(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_serialize(x) for x in data]
    return data


@dataclass
class StepError:
    """Detailed information about a pipeline step failure."""

    step: PipelineStep
    exc_type: str
    message: str
    traceback: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "step": self.step.value,
            "exc_type": self.exc_type,
            "message": self.message,
            "traceback": self.traceback,
        }


class PipelineExecutionError(RuntimeError):
    """Raised when a generation pipeline step fails with additional context."""

    def __init__(self, step: PipelineStep, original_exception: Exception):
        tb = "".join(
            traceback.format_exception(
                type(original_exception),
                original_exception,
                original_exception.__traceback__,
            )
        )
        self.info = StepError(
            step=step,
            exc_type=type(original_exception).__name__,
            message=str(original_exception),
            traceback=tb,
        )
        self.step = step
        self.original_exception = original_exception
        self.traceback = tb
        super().__init__(f"{step.value} failed: {original_exception}")


def load_pipelines_from_file(path: Path) -> Dict[DatasetType, GenerationPipeline]:
    """Load pipeline definitions from a YAML configuration file."""

    import yaml

    data = yaml.safe_load(path.read_text())
    pipelines: Dict[DatasetType, GenerationPipeline] = {}
    for name, info in data.items():
        try:
            dtype = DatasetType(name)
        except ValueError:
            logger.warning("Unknown dataset type in config: %s", name)
            continue
        steps = [PipelineStep(s) for s in info.get("steps", [])]
        trainings = [TrainingGoal(t) for t in info.get("trainings", [])]
        pipelines[dtype] = GenerationPipeline(
            dataset_type=dtype,
            steps=steps,
            compatible_trainings=trainings,
            description=info.get("description", ""),
        )
    return pipelines


DEFAULT_PIPELINES_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "pipelines.yaml"
)
PIPELINE_CONFIG_ENV = "DATACREEK_PIPELINES_CONFIG"


def load_pipelines(path: str | None = None) -> Dict[DatasetType, GenerationPipeline]:
    """Return pipeline definitions from ``path`` or default locations."""

    final_path = Path(path or os.getenv(PIPELINE_CONFIG_ENV, DEFAULT_PIPELINES_PATH))
    if final_path.exists():
        try:
            return load_pipelines_from_file(final_path)
        except Exception:
            logger.exception("Failed to load pipelines from %s", final_path)
    return DEFAULT_PIPELINES


DEFAULT_PIPELINES: Dict[DatasetType, GenerationPipeline] = {
    DatasetType.QA: GenerationPipeline(
        dataset_type=DatasetType.QA,
        steps=[
            PipelineStep.INGEST,
            PipelineStep.TO_KG,
            PipelineStep.VERIFY_THRESHOLDS,
            PipelineStep.KG_CLEANUP,
            PipelineStep.GENERATE_QA,
            PipelineStep.CURATE,
            PipelineStep.SAVE,
        ],
        compatible_trainings=[
            TrainingGoal.SFT,
            TrainingGoal.DPO,
            TrainingGoal.ORPO,
            TrainingGoal.DPO_SFT,
            TrainingGoal.PPO,
            TrainingGoal.RRHF,
            TrainingGoal.RLAIF,
            TrainingGoal.GRPO,
        ],
        description="Question-answer pairs for instruction tuning and preference based training.",
    ),
    DatasetType.COT: GenerationPipeline(
        dataset_type=DatasetType.COT,
        steps=[
            PipelineStep.INGEST,
            PipelineStep.TO_KG,
            PipelineStep.VERIFY_THRESHOLDS,
            PipelineStep.KG_CLEANUP,
            PipelineStep.GENERATE_COT,
            PipelineStep.CURATE,
            PipelineStep.SAVE,
        ],
        compatible_trainings=[
            TrainingGoal.SFT,
            TrainingGoal.DPO,
            TrainingGoal.ORPO,
            TrainingGoal.DPO_SFT,
            TrainingGoal.RRHF,
        ],
        description="Chain-of-thought traces to teach stepwise reasoning.",
    ),
    DatasetType.VQA: GenerationPipeline(
        dataset_type=DatasetType.VQA,
        steps=[
            PipelineStep.INGEST,
            PipelineStep.TO_KG,
            PipelineStep.VERIFY_THRESHOLDS,
            PipelineStep.KG_CLEANUP,
            PipelineStep.GENERATE_VQA,
            PipelineStep.CURATE,
            PipelineStep.SAVE,
        ],
        compatible_trainings=[TrainingGoal.SFT],
        description="Visual question answering pairs.",
    ),
    DatasetType.TEXT: GenerationPipeline(
        dataset_type=DatasetType.TEXT,
        steps=[PipelineStep.INGEST, PipelineStep.TO_KG, PipelineStep.SAVE],
        compatible_trainings=[TrainingGoal.CPT],
        description="Raw text corpus for continual pre-training.",
    ),
    DatasetType.KG: GenerationPipeline(
        dataset_type=DatasetType.KG,
        steps=[
            PipelineStep.INGEST,
            PipelineStep.TO_KG,
            PipelineStep.VERIFY_THRESHOLDS,
            PipelineStep.KG_CLEANUP,
            PipelineStep.GENERATE_FROM_KG,
            PipelineStep.CURATE,
            PipelineStep.SAVE,
        ],
        compatible_trainings=[
            TrainingGoal.SFT,
            TrainingGoal.DPO,
            TrainingGoal.ORPO,
            TrainingGoal.DPO_SFT,
            TrainingGoal.PPO,
            TrainingGoal.RRHF,
            TrainingGoal.RLAIF,
            TrainingGoal.GRPO,
        ],
        description="Question answering data generated from a knowledge graph.",
    ),
    DatasetType.PREF_PAIR: GenerationPipeline(
        dataset_type=DatasetType.PREF_PAIR,
        steps=[
            PipelineStep.INGEST,
            PipelineStep.TO_KG,
            PipelineStep.VERIFY_THRESHOLDS,
            PipelineStep.KG_CLEANUP,
            PipelineStep.GENERATE_CANDIDATES,
            PipelineStep.LABEL_PAIRS,
            PipelineStep.SAVE,
        ],
        compatible_trainings=[
            TrainingGoal.PPO,
            TrainingGoal.DPO,
            TrainingGoal.ORPO,
            TrainingGoal.DPO_SFT,
            TrainingGoal.RLAIF,
        ],
        description="Pairwise preferences for reward-model and preference-based training.",
    ),
    DatasetType.PREF_LIST: GenerationPipeline(
        dataset_type=DatasetType.PREF_LIST,
        steps=[
            PipelineStep.INGEST,
            PipelineStep.TO_KG,
            PipelineStep.VERIFY_THRESHOLDS,
            PipelineStep.KG_CLEANUP,
            PipelineStep.GENERATE_CANDIDATES,
            PipelineStep.RANK_RESPONSES,
            PipelineStep.SAVE,
        ],
        compatible_trainings=[TrainingGoal.GRPO, TrainingGoal.RRHF],
        description="Listwise ranked responses for GRPO or RRHF.",
    ),
    DatasetType.TOOL: GenerationPipeline(
        dataset_type=DatasetType.TOOL,
        steps=[
            PipelineStep.INGEST,
            PipelineStep.TO_KG,
            PipelineStep.VERIFY_THRESHOLDS,
            PipelineStep.KG_CLEANUP,
            PipelineStep.GENERATE_TOOL_CALL,
            PipelineStep.CURATE,
            PipelineStep.SAVE,
        ],
        compatible_trainings=[
            TrainingGoal.SFT,
            TrainingGoal.DPO,
            TrainingGoal.ORPO,
            TrainingGoal.DPO_SFT,
            TrainingGoal.PPO,
            TrainingGoal.RRHF,
            TrainingGoal.RLAIF,
            TrainingGoal.GRPO,
        ],
        description="Single tool-calling demonstrations with integrated results.",
    ),
    DatasetType.CONVERSATION: GenerationPipeline(
        dataset_type=DatasetType.CONVERSATION,
        steps=[
            PipelineStep.INGEST,
            PipelineStep.TO_KG,
            PipelineStep.VERIFY_THRESHOLDS,
            PipelineStep.KG_CLEANUP,
            PipelineStep.GENERATE_CONVERSATION,
            PipelineStep.CURATE,
            PipelineStep.SAVE,
        ],
        compatible_trainings=[
            TrainingGoal.SFT,
            TrainingGoal.DPO,
            TrainingGoal.ORPO,
            TrainingGoal.DPO_SFT,
            TrainingGoal.PPO,
            TrainingGoal.RRHF,
            TrainingGoal.RLAIF,
            TrainingGoal.GRPO,
        ],
        description="Multi-turn conversations for dialogue training.",
    ),
    DatasetType.MULTI_TOOL: GenerationPipeline(
        dataset_type=DatasetType.MULTI_TOOL,
        steps=[
            PipelineStep.INGEST,
            PipelineStep.TO_KG,
            PipelineStep.VERIFY_THRESHOLDS,
            PipelineStep.KG_CLEANUP,
            PipelineStep.GENERATE_MULTI_TOOL,
            PipelineStep.CURATE,
            PipelineStep.SAVE,
        ],
        compatible_trainings=[
            TrainingGoal.SFT,
            TrainingGoal.DPO,
            TrainingGoal.ORPO,
            TrainingGoal.DPO_SFT,
            TrainingGoal.PPO,
            TrainingGoal.RRHF,
            TrainingGoal.RLAIF,
            TrainingGoal.GRPO,
        ],
        description="Sequential multi-tool use traces for complex tasks.",
    ),
}

PIPELINES = load_pipelines()


def get_pipeline(dataset_type: DatasetType) -> GenerationPipeline:
    """Return pipeline information for a dataset type."""

    pipelines = load_pipelines()
    if dataset_type not in pipelines:
        raise KeyError(f"Unknown dataset type: {dataset_type}")
    return pipelines[dataset_type]


def get_trainings_for_dataset(dataset_type: DatasetType) -> List[TrainingGoal]:
    """Return compatible trainings for a dataset type."""

    return get_pipeline(dataset_type).compatible_trainings


def get_dataset_types_for_training(goal: TrainingGoal) -> List[DatasetType]:
    """Return dataset types that can be used for the given training goal."""

    return [
        pipeline.dataset_type
        for pipeline in load_pipelines().values()
        if goal in pipeline.compatible_trainings
    ]


def get_pipelines_for_training(goal: TrainingGoal) -> List[GenerationPipeline]:
    """Return generation pipelines compatible with the given training goal."""

    return [
        pipeline
        for pipeline in load_pipelines().values()
        if goal in pipeline.compatible_trainings
    ]


async def _run_generation_pipeline_impl(  # pragma: no cover - integration helper
    dataset_type: DatasetType,
    kg: KnowledgeGraph,
    *,
    dataset_builder: "DatasetBuilder | None" = None,
    pipeline_config_path: Path | None = None,
    config_path: Path | None = None,
    provider: str | None = None,
    profile: str | None = None,
    api_base: str | None = None,
    model: str | None = None,
    num_pairs: int | None = None,
    threshold: float | None = None,
    fmt: str | None = None,
    overrides: Dict[str, Any] | None = None,
    verbose: bool = False,
    async_mode: bool = False,
    document_text: str | None = None,
    multi_answer: bool = False,
    resolve_threshold: float = 0.8,
    resolve_aliases: dict[str, list[str]] | None = None,
    use_async_handlers: bool = False,
    batch_size: int | None = None,
    inference_batch: int | None = None,
    start_step: PipelineStep | None = None,
    dedup_similarity: float = 1.0,
    keep_ratings: bool = False,
    curation_temperature: float | None = None,
    resume_curation: bool = False,
    redis_client: redis.Redis | None = None,
) -> Any:
    """Shared implementation for sync and async generation pipelines."""

    try:
        if pipeline_config_path:
            pipelines = load_pipelines_from_file(pipeline_config_path)
            pipeline = pipelines[dataset_type]
        else:
            pipeline = load_pipelines().get(dataset_type) or get_pipeline(dataset_type)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc

    # The TEXT dataset does not include generation after the knowledge graph,
    # so it remains unsupported by this helper.
    if dataset_type == DatasetType.TEXT:
        raise ValueError(
            "run_generation_pipeline does not support the TEXT dataset type"
        )

    if document_text is None:
        document_text = kg.to_text()

    cfg = load_config_with_overrides(
        str(config_path) if config_path else None, overrides
    )
    fmt_cfg = get_format_settings(cfg)

    options = ProcessOptions(
        config_path=config_path,
        provider=provider,
        profile=profile,
        api_base=api_base,
        model=model,
        num_pairs=num_pairs,
        overrides=overrides,
        verbose=verbose,
        async_mode=async_mode,
        kg=kg,
        multi_answer=multi_answer,
        batch_size=batch_size,
        inference_batch=inference_batch,
        keep_ratings=keep_ratings,
        dedup_similarity=dedup_similarity,
        curation_temperature=curation_temperature,
        curation_threshold=threshold,
        resume_curation=resume_curation,
    )
    if redis_client is None:
        if dataset_builder and dataset_builder.redis_client:
            redis_client = dataset_builder.redis_client
        else:
            redis_client = get_redis_client()

    cp_key = None
    if redis_client and dataset_builder and dataset_builder.name:
        cp_key = f"dataset:{dataset_builder.name}:progress"
        if start_step is None:
            meta = {
                "dataset_type": dataset_type.value,
                "model": model,
                "dedup_similarity": dedup_similarity,
                "keep_ratings": keep_ratings,
            }
            redis_client.hset(cp_key, "metadata", json.dumps(meta))

    data: Any = document_text
    if start_step and cp_key:
        idx = pipeline.steps.index(start_step)
        if idx > 0:
            prev_step = pipeline.steps[idx - 1]
            stored = redis_client.hget(cp_key, prev_step.value)
            if stored:
                data = json.loads(stored)

    async def _identity(d: Any) -> Any:
        return d

    async def _save(d: Any) -> Any:
        format_type = fmt or fmt_cfg.default
        return await asyncio.to_thread(convert_format, _serialize(d), format_type, cfg)

    async def _generate(ct: ContentType, text: str) -> Any:
        if use_async_handlers:
            return await process_file_async(
                None,
                None,
                options.config_path,
                options.api_base,
                options.model,
                ct,
                options.num_pairs,
                options.verbose,
                provider=options.provider,
                profile=options.profile,
                document_text=text,
                kg=options.kg,
                config_overrides=options.overrides,
                multi_answer=(
                    options.multi_answer if ct is ContentType.FROM_KG else False
                ),
            )
        return await asyncio.to_thread(
            process_file,
            None,
            None,
            options.config_path,
            options.api_base,
            options.model,
            ct,
            options.num_pairs,
            options.verbose,
            async_mode=options.async_mode,
            provider=options.provider,
            profile=options.profile,
            document_text=text,
            kg=options.kg,
            config_overrides=options.overrides,
            multi_answer=options.multi_answer if ct is ContentType.FROM_KG else False,
        )

    async def _curate(d: Any) -> Any:
        if use_async_handlers:
            result = await curate_qa_pairs_async(
                d,
                None,
                threshold,
                options.api_base,
                options.model,
                options.config_path,
                options.verbose,
                options.provider,
                kg=options.kg,
                batch_size=options.batch_size,
                inference_batch=options.inference_batch,
                keep_ratings=options.keep_ratings,
                temperature=options.curation_temperature,
                resume=options.resume_curation,
                as_dataclass=True,
            )
        else:
            result = await asyncio.to_thread(
                curate_qa_pairs,
                d,
                None,
                threshold,
                options.api_base,
                options.model,
                options.config_path,
                options.verbose,
                options.provider,
                async_mode=options.async_mode,
                kg=options.kg,
                batch_size=options.batch_size,
                inference_batch=options.inference_batch,
                keep_ratings=options.keep_ratings,
                temperature=options.curation_temperature,
                resume=options.resume_curation,
                as_dataclass=True,
            )

        metrics = None
        if isinstance(result, dict):
            metrics = result.get("metrics")
        elif hasattr(result, "metrics"):
            metrics = result.metrics
        if dataset_builder is not None and metrics is not None:
            metrics_dict = asdict(metrics) if is_dataclass(metrics) else metrics
            dataset_builder._record_event(
                "curate",
                "Curated %d/%d QA pairs"
                % (
                    metrics_dict.get("filtered", 0),
                    metrics_dict.get("total", 0),
                ),
                **metrics_dict,
            )

        # Convert plain dictionaries to a CurationResult for consistency
        if isinstance(result, dict):
            qa_pairs = [
                QAPair(**p) if isinstance(p, dict) else p
                for p in result.get("qa_pairs", [])
            ]
            rated_pairs = (
                [QAPair(**p) for p in result.get("rated_pairs", [])]
                if "rated_pairs" in result
                else None
            )
            metrics_obj = metrics if isinstance(metrics, CurationMetrics) else None
            if metrics_obj is None:
                metrics_obj = CurationMetrics(
                    total=len(qa_pairs),
                    filtered=len(qa_pairs),
                    retention_rate=1.0,
                    avg_score=0.0,
                )
            result = CurationResult(
                summary=result.get("summary", ""),
                qa_pairs=qa_pairs,
                conversations=result.get("conversations", []),
                metrics=metrics_obj,
                rated_pairs=rated_pairs,
            )

        return result

    async def _kg_cleanup(d: Any) -> Any:
        stats = await asyncio.to_thread(
            cleanup_knowledge_graph,
            kg,
            dataset_builder=dataset_builder,
            resolve_threshold=resolve_threshold,
            resolve_aliases=resolve_aliases,
            dedup_similarity=dedup_similarity,
        )
        if verbose:
            logger.info(
                "  KG cleanup - removed:%d cleaned:%d", stats.removed, stats.cleaned
            )
        return d

    handlers = {
        PipelineStep.GENERATE_QA: lambda d: _generate(ContentType.QA, d),
        PipelineStep.GENERATE_COT: lambda d: _generate(ContentType.COT, d),
        PipelineStep.GENERATE_VQA: lambda d: _generate(
            ContentType.VQA_ADD_REASONING, d
        ),
        PipelineStep.GENERATE_FROM_KG: lambda d: _generate(ContentType.FROM_KG, d),
        PipelineStep.GENERATE_TOOL_CALL: lambda d: _generate(ContentType.TOOL_CALL, d),
        PipelineStep.GENERATE_CONVERSATION: lambda d: _generate(
            ContentType.CONVERSATION, d
        ),
        PipelineStep.GENERATE_MULTI_TOOL: lambda d: _generate(
            ContentType.MULTI_TOOL, d
        ),
        PipelineStep.GENERATE_CANDIDATES: lambda d: _generate(
            (
                ContentType.PREF_PAIR
                if dataset_type == DatasetType.PREF_PAIR
                else ContentType.PREF_LIST
            ),
            d,
        ),
        PipelineStep.LABEL_PAIRS: _identity,
        PipelineStep.RANK_RESPONSES: _identity,
        PipelineStep.VERIFY_THRESHOLDS: lambda d: (verify_thresholds() or d),
        PipelineStep.KG_CLEANUP: _kg_cleanup,
        PipelineStep.CURATE: _curate,
        PipelineStep.SAVE: _save,
    }

    start_idx = 0
    if start_step and start_step in pipeline.steps:
        start_idx = pipeline.steps.index(start_step)

    exec_steps = [
        s
        for s in pipeline.steps[start_idx:]
        if s not in {PipelineStep.INGEST, PipelineStep.TO_KG}
    ]
    if start_step is PipelineStep.CURATE and PipelineStep.SAVE in exec_steps:
        exec_steps.remove(PipelineStep.SAVE)
    progress = None
    task = None
    if verbose:
        progress, task = create_progress("Generation pipeline", len(exec_steps))
        progress.start()

    try:
        for idx, step in enumerate(exec_steps):
            if step in {PipelineStep.INGEST, PipelineStep.TO_KG}:
                continue
            if verbose:
                logger.info("Running step %s", step.value)
            handler = handlers.get(step)
            if handler:
                if cp_key:
                    redis_client.hset(
                        cp_key,
                        f"{step.value}_start",
                        datetime.now(timezone.utc).isoformat(),
                    )
                start = time.perf_counter()
                try:
                    result = await handler(data)
                except Exception as exc:
                    logger.exception("Step %s failed", step.value)
                    raise PipelineExecutionError(step, exc) from exc
                duration = time.perf_counter() - start
                if step.name.startswith("GENERATE"):
                    data = _validate_step_result(dataset_type, step, result)
                else:
                    data = result
                if cp_key:
                    redis_client.hset(cp_key, step.value, json.dumps(_serialize(data)))
                    redis_client.hset(
                        cp_key,
                        f"{step.value}_duration",
                        f"{duration:.2f}",
                    )
                    progress_val = round((idx + 1) / len(exec_steps), 3)
                    redis_client.hset(cp_key, "progress", progress_val)
                    redis_client.hset(
                        cp_key,
                        f"{step.value}_finish",
                        datetime.now(timezone.utc).isoformat(),
                    )
                if verbose:
                    logger.info("Finished %s in %.2fs", step.value, duration)
                if progress:
                    progress.update(task, advance=1)
                if dataset_builder is not None:
                    try:
                        dataset_builder._record_event(
                            step.value,
                            f"{step.value} completed",
                            duration=f"{duration:.2f}s",
                        )
                    except Exception:
                        logger.exception("Failed to record event for %s", step.value)
                if isinstance(data, dict) or is_dataclass(data):
                    info = asdict(data) if is_dataclass(data) else data
                    if "qa_pairs" in info:
                        logger.info("  Pairs: %d", len(info["qa_pairs"]))
                    if "conversations" in info:
                        logger.info("  Conversations: %d", len(info["conversations"]))
                    if step is PipelineStep.CURATE and "metrics" in info:
                        m = info["metrics"]
                        if is_dataclass(m):
                            m = asdict(m)
                        logger.info(
                            "  Curation metrics - total:%d filtered:%d retention:%.2f avg:%.1f",
                            m.get("total", 0),
                            m.get("filtered", 0),
                            m.get("retention_rate", 0.0),
                            m.get("avg_score", 0.0),
                        )
    finally:
        if progress:
            progress.stop()

    return data


def run_generation_pipeline(  # pragma: no cover - integration helper
    dataset_type: DatasetType,
    kg: KnowledgeGraph,
    *,
    dataset_builder: "DatasetBuilder | None" = None,
    pipeline_config_path: Path | None = None,
    batch_size: int | None = None,
    inference_batch: int | None = None,
    dedup_similarity: float = 1.0,
    keep_ratings: bool = False,
    curation_threshold: float | None = None,
    curation_temperature: float | None = None,
    resume_curation: bool = False,
    resolve_threshold: float = 0.8,
    resolve_aliases: dict[str, list[str]] | None = None,
    start_step: PipelineStep | None = None,
    redis_client: redis.Redis | None = None,
    **kwargs: Any,
) -> Any:
    """Execute the generation steps synchronously.

    Parameters
    ----------
    dataset_builder:
        If provided, knowledge graph cleanup will log events on this builder.
    pipeline_config_path:
        Optional path to a YAML file describing pipeline definitions.
    pipeline_config_path:
        Optional path to a YAML file describing pipeline definitions.
    dedup_similarity:
        Similarity used when removing duplicate chunks during KG cleanup.
    keep_ratings:
        Return ratings for all generated pairs after curation.
    curation_threshold:
        Override the quality threshold used during curation.
    curation_temperature:
        Override the temperature used when rating pairs during curation.
    resume_curation:
        Continue curation if previous ratings exist by loading progress from Redis.
    resolve_threshold:
        Similarity threshold used when merging entities during knowledge graph
        cleanup.
    resolve_aliases:
        Optional alias mapping passed to :meth:`DatasetBuilder.resolve_entities`.

    Raises
    ------
    PipelineExecutionError
        If any step fails.
    """

    return asyncio.run(
        _run_generation_pipeline_impl(
            dataset_type,
            kg,
            use_async_handlers=False,
            dataset_builder=dataset_builder,
            pipeline_config_path=pipeline_config_path,
            batch_size=batch_size,
            inference_batch=inference_batch,
            dedup_similarity=dedup_similarity,
            keep_ratings=keep_ratings,
            threshold=curation_threshold,
            curation_temperature=curation_temperature,
            resume_curation=resume_curation,
            resolve_threshold=resolve_threshold,
            resolve_aliases=resolve_aliases,
            start_step=start_step,
            redis_client=redis_client,
            **kwargs,
        )
    )


async def run_generation_pipeline_async(  # pragma: no cover - integration helper
    dataset_type: DatasetType,
    kg: KnowledgeGraph,
    *,
    dataset_builder: "DatasetBuilder | None" = None,
    pipeline_config_path: Path | None = None,
    batch_size: int | None = None,
    inference_batch: int | None = None,
    dedup_similarity: float = 1.0,
    keep_ratings: bool = False,
    curation_threshold: float | None = None,
    curation_temperature: float | None = None,
    resume_curation: bool = False,
    resolve_threshold: float = 0.8,
    resolve_aliases: dict[str, list[str]] | None = None,
    start_step: PipelineStep | None = None,
    redis_client: redis.Redis | None = None,
    **kwargs: Any,
) -> Any:
    """Asynchronous counterpart to :func:`run_generation_pipeline`.

    Parameters
    ----------
    dataset_builder:
        If provided, knowledge graph cleanup will log events on this builder.
    dedup_similarity:
        Similarity used when removing duplicate chunks during KG cleanup.
    keep_ratings:
        Return ratings for all generated pairs after curation.
    curation_threshold:
        Override the quality threshold used during curation.
    curation_temperature:
        Override the temperature used when rating pairs during curation.
    resume_curation:
        Continue curation if previous ratings exist by loading progress from Redis.
    resolve_threshold:
        Similarity threshold used when merging entities during knowledge graph
        cleanup.
    resolve_aliases:
        Optional alias mapping passed to :meth:`DatasetBuilder.resolve_entities`.

    Raises
    ------
    PipelineExecutionError
        If any step fails.
    """

    kwargs["async_mode"] = True
    return await _run_generation_pipeline_impl(
        dataset_type,
        kg,
        use_async_handlers=True,
        dataset_builder=dataset_builder,
        pipeline_config_path=pipeline_config_path,
        batch_size=batch_size,
        inference_batch=inference_batch,
        dedup_similarity=dedup_similarity,
        keep_ratings=keep_ratings,
        threshold=curation_threshold,
        curation_temperature=curation_temperature,
        resume_curation=resume_curation,
        resolve_threshold=resolve_threshold,
        resolve_aliases=resolve_aliases,
        start_step=start_step,
        redis_client=redis_client,
        **kwargs,
    )
