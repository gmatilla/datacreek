from __future__ import annotations

"""Image captioning utilities."""

import logging
import time
import importlib
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Iterable, List

LOGGER = logging.getLogger(__name__)
_CAPTIONING_ERR: str | None = None


@lru_cache(maxsize=1)
def _get_model():
    """Return a cached BLIP captioning pipeline."""
    global _CAPTIONING_ERR
    if _CAPTIONING_ERR:
        raise RuntimeError(_CAPTIONING_ERR)
    try:
        transformers = importlib.import_module("transformers")
        pipeline = getattr(transformers, "pipeline")
    except Exception as exc:  # pragma: no cover - optional dependency missing
        _CAPTIONING_ERR = (
            "transformers (with vision extras) is required for image captioning"
        )
        raise RuntimeError(_CAPTIONING_ERR) from exc

    try:
        return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    except Exception as exc:  # pragma: no cover - runtime errors
        _CAPTIONING_ERR = f"Failed to instantiate BLIP pipeline: {exc}"
        raise RuntimeError(_CAPTIONING_ERR) from exc


def caption_image(path: str) -> str:
    """Return a caption for the image at ``path``."""
    try:
        model = _get_model()
    except RuntimeError as exc:
        LOGGER.warning("image captioning unavailable: %s", exc)
        return ""
    try:
        result = model(path)
    except Exception as exc:  # pragma: no cover - runtime errors
        LOGGER.exception("Failed to caption image %s", path)
        return ""
    if not result:
        return ""
    data = result[0]
    return data.get("generated_text") or data.get("caption", "")


def caption_images_parallel(
    paths: Iterable[str],
    *,
    max_workers: int | None = None,
    chunk_size: int = 256,
) -> List[str]:
    """Return BLIP captions for ``paths`` using a thread pool.

    Parameters
    ----------
    paths:
        Iterable of image file paths to caption.
    max_workers:
        Maximum number of worker threads. Defaults to ``None`` which
        lets :class:`~concurrent.futures.ThreadPoolExecutor` decide.
    chunk_size:
        Number of images processed per pool. Default is ``256`` as
        specified in the scale-out roadmap.

    Returns
    -------
    List[str]
        Captions for each image in ``paths`` in order.
    """

    start = time.perf_counter()
    all_paths = list(paths)
    captions: List[str] = []
    for i in range(0, len(all_paths), chunk_size):
        chunk = all_paths[i : i + chunk_size]
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            captions.extend(list(ex.map(caption_image, chunk)))
    duration = time.perf_counter() - start
    if duration > 0:
        rate = len(all_paths) / duration
    else:  # pragma: no cover - extremely fast
        rate = float("inf")
    logging.getLogger(__name__).debug(
        "BLIP parallel captioning %d images in %.2fs (%.2f img/s)",
        len(all_paths),
        duration,
        rate,
    )
    return captions
