"""Image quality scoring utilities."""

from __future__ import annotations

import numpy as np
from PIL import Image

__all__ = ["sharpness_exposure", "should_caption"]


def sharpness_exposure(path: str) -> tuple[float, float]:
    """Return (sharpness, exposure) metrics for the image at ``path``.

    Sharpness is derived from the variance of the Laplacian using a
    five-point stencil. Exposure measures closeness of the mean luminance
    to mid-gray. Both metrics are scaled to ``[0, 1]``.
    """
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=float) / 255.0
    img.close()

    lap = (
        -4 * arr
        + np.roll(arr, 1, 0)
        + np.roll(arr, -1, 0)
        + np.roll(arr, 1, 1)
        + np.roll(arr, -1, 1)
    )
    sharp = float(lap.var() / 10.0)
    mean = float(arr.mean())
    exposure = float(1.0 - abs(mean - 0.5) * 2)
    return sharp, exposure


def should_caption(path: str, threshold: float = 0.4) -> bool:
    r"""Return ``True`` if the image should be captioned with BLIP.

    The quality metric :math:`Q=\sqrt{sharpness\cdot exposure}` must
    exceed ``threshold`` to trigger captioning.
    """
    sharp, exposure = sharpness_exposure(path)
    quality = (sharp * exposure) ** 0.5
    return quality > threshold
