r"""Pydantic schemas used for ingestion payload validation.

The models include basic path checking and optional quality metrics used as
"quality gates" during ingestion. Metrics follow:

* ``blur_score`` for images based on the variance of the Laplacian of the image
  :math:`\mathrm{Blur} = \tfrac{1}{|I|}\sum (\nabla^2 I)^2`.
* ``entropy`` for text/PDF measured in bits/character
  :math:`\mathrm{Entropy} = -\sum p_i \log_2 p_i`.

These gates ensure minimal data quality before a file enters the pipeline.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    constr,
    field_validator,
)


class BaseIngest(BaseModel):
    """Base schema for ingest payloads."""

    path: constr(min_length=1)

    model_config = ConfigDict(extra="forbid")


class ImageIngest(BaseIngest):
    """Schema for image ingestion payloads."""

    # Dimensions in pixels with minimal acceptable values
    width: int | None = Field(
        default=None, ge=256, description="Image width in pixels (>=256)"
    )
    height: int | None = Field(
        default=None, ge=256, description="Image height in pixels (>=256)"
    )
    # Variance of Laplacian for blur detection
    blur_score: float | None = Field(
        default=None, le=0.2, description="Blur metric; lower is sharper"
    )
    high_res: bool = False


class AudioIngest(BaseIngest):
    """Schema for audio ingestion payloads."""

    sample_rate: int | None = None
    # Duration in seconds (max 4 hours)
    duration: float | None = Field(
        default=None, le=4 * 3600, description="Audio duration in seconds"
    )
    # Signal-to-noise ratio in decibels
    snr: float | None = Field(
        default=None, ge=10.0, description="Signal-to-noise ratio (dB)"
    )


class PdfIngest(BaseIngest):
    """Schema for PDF ingestion payloads."""

    ocr: bool = False
    # Shannon entropy of text content (bits/char)
    entropy: float | None = Field(
        default=None, ge=3.5, description="Content entropy in bits/char"
    )
