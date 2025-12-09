"""Utilities for loading Unsloth models and applying PEFT adapters.

This module wraps `unsloth.FastLanguageModel` to provide a thin
abstraction for loading models in a memory-efficient manner and
attaching LoRA adapters via PEFT. The functions are intentionally
lightweight so that they can be easily mocked during tests.
"""

from __future__ import annotations

from typing import Any, Iterable

try:
    # Import is optional at runtime to allow mocking during tests.
    from unsloth import FastLanguageModel  # type: ignore
except Exception as exc:  # pragma: no cover - executed only when import fails.
    FastLanguageModel = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:  # pragma: no cover - only executed when import succeeds.
    _IMPORT_ERROR = None


def load_model(model_id: str, bits: int = 4, max_seq: int = 8192, **kwargs: Any) -> Any:
    """Load a model using :class:`unsloth.FastLanguageModel`.

    Parameters
    ----------
    model_id:
        Identifier of the pretrained model to download.
    bits:
        Number of quantization bits. Defaults to 4 for 4-bit loading.
    max_seq:
        Maximum sequence length for the model.
    kwargs:
        Additional keyword arguments forwarded to
        :meth:`FastLanguageModel.from_pretrained`.

    Returns
    -------
    Any
        The instantiated model.
    """

    if FastLanguageModel is None:  # pragma: no cover - safeguard when unsloth missing.
        raise ImportError("unsloth is required to load models") from _IMPORT_ERROR

    return FastLanguageModel.from_pretrained(
        model_id,
        load_in_4bit=bits == 4,
        max_seq_len=max_seq,
        **kwargs,
    )


def add_lora(
    model: Any,
    r: int,
    alpha: int,
    target_modules: Iterable[str],
    **kwargs: Any,
) -> Any:
    """Attach LoRA adapters to a model.

    Parameters
    ----------
    model:
        The base model returned by :func:`load_model`.
    r:
        Rank of the LoRA decomposition.
    alpha:
        Scaling factor applied to the LoRA updates.
    target_modules:
        Iterable of module names to which adapters should be applied.
    kwargs:
        Additional keyword arguments forwarded to
        :meth:`FastLanguageModel.get_peft_model`.

    Returns
    -------
    Any
        Model wrapped with PEFT adapters.
    """

    if FastLanguageModel is None:  # pragma: no cover - safeguard when unsloth missing.
        raise ImportError("unsloth is required to apply LoRA") from _IMPORT_ERROR

    return FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        **kwargs,
    )
