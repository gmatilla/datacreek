from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:  # optional dependency
    from pydantic import BaseModel
except Exception:  # pragma: no cover - fallback when pydantic missing

    class BaseModel:  # lightweight stub mimicking pydantic model
        """Minimal replacement for :class:`pydantic.BaseModel`."""

        def __init__(self, **data: Any) -> None:  # type: ignore[override]
            for key, value in data.items():
                setattr(self, key, value)

        @classmethod
        def model_validate(cls, data: Dict[str, Any]):
            return cls(**data)

        def model_dump(self) -> Dict[str, Any]:
            return self.__dict__


@dataclass
class GenerationSettings:
    """Configuration options controlling LLM generation."""

    temperature: float = 0.7
    top_p: float = 0.95
    chunk_size: int = 4000
    overlap: int = 200
    # basic|sliding|semantic|contextual|summary
    chunk_method: str = "sliding"
    similarity_drop: float = 0.3
    retrieval_top_k: int = 3
    max_tokens: int = 4096
    num_pairs: int = 25
    num_cot_examples: int = 5
    num_cot_enhance_examples: Optional[int] = None
    batch_size: int = 32
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = field(default=None)
    summary_temperature: float = 0.1
    summary_max_tokens: int = 1024

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationSettings":
        """Create ``GenerationSettings`` from a raw dictionary."""
        defaults = cls()
        return cls(
            **{k: data.get(k, getattr(defaults, k)) for k in cls.__dataclass_fields__}
        )

    def update(self, overrides: Dict[str, Any]) -> None:
        """Update fields from a dictionary of overrides."""
        for key, value in overrides.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)


class GenerationSettingsModel(BaseModel):
    """Pydantic model for validating generation settings."""

    temperature: float = 0.7
    top_p: float = 0.95
    chunk_size: int = 4000
    overlap: int = 200
    # basic|sliding|semantic|contextual|summary
    chunk_method: str = "sliding"
    similarity_drop: float = 0.3
    retrieval_top_k: int = 3
    max_tokens: int = 4096
    num_pairs: int = 25
    num_cot_examples: int = 5
    num_cot_enhance_examples: Optional[int] = None
    batch_size: int = 32
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    summary_temperature: float = 0.1
    summary_max_tokens: int = 1024

    def to_settings(self) -> GenerationSettings:
        """Convert to :class:`GenerationSettings`."""
        return GenerationSettings.from_dict(self.model_dump())


@dataclass
class CurateSettings:
    """Configuration for curation and rating of generated data."""

    threshold: float = 7.0
    batch_size: int = 32
    inference_batch: int = 32
    temperature: float = 0.1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CurateSettings":
        defaults = cls()
        return cls(
            **{k: data.get(k, getattr(defaults, k)) for k in cls.__dataclass_fields__}
        )


class CurateSettingsModel(BaseModel):
    threshold: float = 7.0
    batch_size: int = 32
    inference_batch: int = 32
    temperature: float = 0.1

    def to_settings(self) -> CurateSettings:
        return CurateSettings.from_dict(self.model_dump())


@dataclass
class FormatSettings:
    """Configuration for output formatting."""

    default: str = "jsonl"
    include_metadata: bool = True
    pretty_json: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FormatSettings":
        defaults = cls()
        return cls(
            **{k: data.get(k, getattr(defaults, k)) for k in cls.__dataclass_fields__}
        )


class FormatSettingsModel(BaseModel):
    default: str = "jsonl"
    include_metadata: bool = True
    pretty_json: bool = True

    def to_settings(self) -> FormatSettings:
        return FormatSettings.from_dict(self.model_dump())


@dataclass
class VLLMSettings:
    """Configuration for vLLM server connection."""

    api_base: str = "http://localhost:8000/v1"
    port: int = 8000
    model: str = "meta-llama/Llama-3.3-70B-Instruct"
    max_retries: int = 3
    retry_delay: float = 1.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VLLMSettings":
        defaults = cls()
        return cls(
            **{k: data.get(k, getattr(defaults, k)) for k in cls.__dataclass_fields__}
        )


class VLLMSettingsModel(BaseModel):
    api_base: str = "http://localhost:8000/v1"
    port: int = 8000
    model: str = "meta-llama/Llama-3.3-70B-Instruct"
    max_retries: int = 3
    retry_delay: float = 1.0

    def to_settings(self) -> VLLMSettings:
        return VLLMSettings.from_dict(self.model_dump())


@dataclass
class OpenAISettings:
    """Configuration for OpenAI or compatible API endpoint."""

    api_base: Optional[str] = None
    api_key: Optional[str] = None
    model: str = "gpt-4o"
    max_retries: int = 3
    retry_delay: float = 1.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenAISettings":
        defaults = cls()
        return cls(
            **{k: data.get(k, getattr(defaults, k)) for k in cls.__dataclass_fields__}
        )


class OpenAISettingsModel(BaseModel):
    api_base: str | None = None
    api_key: str | None = None
    model: str = "gpt-4o"
    max_retries: int = 3
    retry_delay: float = 1.0

    def to_settings(self) -> OpenAISettings:
        return OpenAISettings.from_dict(self.model_dump())


@dataclass
class LLMSettings:
    """General LLM configuration selecting the provider."""

    provider: str = "vllm"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMSettings":
        defaults = cls()
        return cls(
            **{k: data.get(k, getattr(defaults, k)) for k in cls.__dataclass_fields__}
        )


class LLMSettingsModel(BaseModel):
    provider: str = "vllm"

    def to_settings(self) -> LLMSettings:
        return LLMSettings.from_dict(self.model_dump())
