from __future__ import annotations

from typing import Iterable, List

from .llm_client import LLMClient


class LLMService:
    """Convenient wrapper around :class:`LLMClient`.

    The helper simplifies synchronous and asynchronous completion calls. It can
    be used directly as a callable for single prompts or via the ``batch`` and
    ``abatch`` helpers for (a)synchronous multi-prompt processing.

    Parameters
    ----------
    provider:
        Identifier understood by :class:`LLMClient` selecting the backend
        implementation (e.g. ``"vllm"`` or ``"api-endpoint"``).
    profile:
        Optional profile name when multiple configurations are available.
    api_base:
        Custom API base URL for remote backends.
    model:
        Model or engine name passed through to the client.
    """

    def __init__(
        self,
        *,
        provider: str = "vllm",
        profile: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
    ) -> None:
        self.client = LLMClient(
            provider=provider,
            profile=profile,
            api_base=api_base,
            model_name=model,
        )

    def __call__(self, prompt: str) -> str:
        """Synchronously return a completion for ``prompt``."""

        messages = [{"role": "user", "content": prompt}]
        return self.client.chat_completion(messages)

    def batch(self, prompts: Iterable[str]) -> List[str]:
        """Return completions for many ``prompts`` in a blocking manner."""

        batches = [[{"role": "user", "content": p}] for p in prompts]
        return self.client.batch_completion(batches)

    async def acomplete(self, prompts: Iterable[str]) -> List[str]:
        """Asynchronously complete several ``prompts`` and return the results."""

        batches = [[{"role": "user", "content": p}] for p in prompts]
        return await self.client.async_batch_completion(batches)

    async def abatch(self, prompts: Iterable[str]) -> List[str]:
        """Alias for :meth:`acomplete` for API symmetry."""

        return await self.acomplete(prompts)
