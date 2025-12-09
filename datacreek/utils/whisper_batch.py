"""Batch Whisper.cpp transcription utilities."""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Iterable, List

try:  # optional 8-bit matmul for CPU acceleration
    import bitsandbytes.functional as bnb_fn  # type: ignore
except Exception:  # pragma: no cover - dependency missing
    bnb_fn = None  # type: ignore

# Avoid importing torch on unsupported platforms to prevent crashes
torch = None  # default to disabled
try:  # optional dependency (disabled by default)
    if False:
        import torch  # pragma: no cover
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:  # optional heavy dependency
    from whispercpp import Whisper  # type: ignore
except Exception:  # pragma: no cover - dependency missing
    Whisper = None  # type: ignore

__all__ = ["transcribe_audio_batch"]


@lru_cache(maxsize=1)
def _get_model(
    model: str = "tiny.en",
    fp16: bool = True,
    device: str | None = None,
    *,
    int8: bool = False,
):
    """Return a cached whisper.cpp model instance."""

    if Whisper is None:  # pragma: no cover - dependency missing
        return None
    kwargs = {}
    if int8:
        kwargs["compute_type"] = "int8"
    try:
        return Whisper(model, fp16=fp16, device=device, **kwargs)
    except TypeError:  # pragma: no cover - older bindings
        return Whisper(model, fp16=fp16, device=device)


def transcribe_audio_batch(
    paths: Iterable[str],
    *,
    model: str = "tiny.en",
    max_seconds: int = 30,
    batch_size: int = 8,
    device: str | None = None,
    quantize: bool = True,
) -> List[str]:
    """Return transcripts for ``paths`` using whisper.cpp in batches.

    Parameters
    ----------
    paths:
        Iterable of audio file paths. Each should contain at most ``max_seconds``
        of audio.
    model:
        Name or path of the whisper.cpp model to load. Defaults to ``"tiny.en"``.
    max_seconds:
        Maximum audio length per item in seconds. Longer files are processed but
        may reduce throughput.
    batch_size:
        Number of audio items processed per batch.
    device:
        Optional compute device (e.g. ``"cuda"``). ``None`` selects automatically.
    quantize:
        Use int8 inference when running on CPU for speed.

    Returns
    -------
    list[str]
        Transcribed text for each audio path. Returns empty strings if
        ``whispercpp`` is unavailable.
    """

    all_paths = list(paths)
    if device is None:
        if torch is not None and getattr(torch.cuda, "is_available", lambda: False)():
            device = "cuda"
        else:
            device = "cpu"
    use_fp16 = device == "cuda"
    if device == "cpu":
        batch_size = max(1, batch_size // 4)
        if torch is not None and bnb_fn is not None:
            orig_matmul = torch.matmul
            torch.matmul = bnb_fn.matmul_8bit  # type: ignore[assignment]
        else:
            orig_matmul = None
    else:
        orig_matmul = None
    model_inst = _get_model(
        model, fp16=use_fp16, device=device, int8=quantize and device == "cpu"
    )
    if model_inst is None:  # pragma: no cover - dependency missing
        logging.getLogger(__name__).warning("whispercpp not available")
        return ["" for _ in all_paths]

    from datacreek.analysis.monitoring import update_metric, whisper_fallback_total

    transcripts: List[str] = []
    start = time.perf_counter()
    idx = 0
    try:
        while idx < len(all_paths):
            chunk = all_paths[idx : idx + batch_size]
            step = len(chunk)
            for path in chunk:
                try:
                    text = model_inst.transcribe(path, max_length=max_seconds)
                except Exception as exc:  # pragma: no cover - runtime error
                    if device == "cuda" and "out of memory" in str(exc).lower():
                        logging.getLogger(__name__).warning("whisper OOM, fallback CPU")
                        if whisper_fallback_total is not None:
                            try:
                                whisper_fallback_total.inc()
                            except Exception:
                                pass
                        # reload on CPU, reduce batch
                        _get_model.cache_clear()
                        device = "cpu"
                        use_fp16 = False
                        batch_size = 1
                        model_inst = _get_model(
                            model, fp16=use_fp16, device="cpu", int8=quantize
                        )
                        text = model_inst.transcribe(path, max_length=max_seconds)
                    else:
                        logging.getLogger(__name__).exception(
                            "failed to transcribe %s", path
                        )
                        text = ""
                transcripts.append(text)
            idx += step
    finally:
        if orig_matmul is not None:
            torch.matmul = orig_matmul
    duration = time.perf_counter() - start
    if duration > 0:
        rate = len(all_paths) / duration
        xrt = duration / (max_seconds * len(all_paths))
    else:  # pragma: no cover - extremely fast
        rate = float("inf")
        xrt = 0.0
    update_metric("whisper_xrt", float(xrt), {"device": device})
    logging.getLogger(__name__).debug(
        "Whisper.cpp transcribed %d files in %.2fs (%.2f audio/s)",
        len(all_paths),
        duration,
        rate,
    )
    return transcripts
