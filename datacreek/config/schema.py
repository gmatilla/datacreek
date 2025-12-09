from __future__ import annotations

from typing import Any, Dict

"""Typed configuration models."""

try:  # optional dependency
    from pydantic import BaseModel, ConfigDict, Field
except Exception:  # pragma: no cover - fallback when pydantic missing

    class BaseModel:  # lightweight stub for missing pydantic
        def __init__(self, **data: Any):
            for k, v in data.items():
                setattr(self, k, v)

        @classmethod
        def model_validate(cls, data: Dict[str, Any]):
            return cls(**data)

        def model_dump(self) -> Dict[str, Any]:
            return self.__dict__

    class ConfigDict(dict):
        pass

    def Field(default: Any = None, *args, **kwargs):
        return default


class PidConfig(BaseModel):
    """PID controller settings."""

    model_config = ConfigDict(extra="ignore")

    Kp: float = Field(0.4, gt=0.0, le=1.0, description="Proportional gain")
    Ki: float = Field(0.05, description="Integral gain")


class GpuConfig(BaseModel):
    """GPU acceleration options."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = Field(False, description="Enable GPU features")


class ConfigSchema(BaseModel):
    """Root configuration schema."""

    model_config = ConfigDict(extra="allow")

    pid: PidConfig = Field(default_factory=PidConfig)
    gpu: GpuConfig = Field(default_factory=GpuConfig)
