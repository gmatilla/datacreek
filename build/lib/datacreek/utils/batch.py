import logging
from typing import Any, Callable, Dict, List

from datacreek.models.llm_client import LLMClient

logger = logging.getLogger(__name__)


def process_batches(
    client: LLMClient,
    message_batches: List[List[Dict[str, str]]],
    *,
    batch_size: int,
    temperature: float,
    parse_fn: Callable[[str], Any],
    raise_on_error: bool = False,
) -> List[Any]:
    """Helper to run LLM requests in batches with optional error propagation."""
    results: List[Any] = []
    for start in range(0, len(message_batches), batch_size):
        end = min(start + batch_size, len(message_batches))
        batch = message_batches[start:end]
        try:
            responses = client.batch_completion(
                batch,
                temperature=temperature,
                batch_size=batch_size,
            )
            for resp in responses:
                try:
                    results.append(parse_fn(resp))
                except Exception as parse_err:
                    logger.error("Failed to parse response: %s", parse_err)
        except Exception as e:
            if raise_on_error:
                raise
            logger.error("Error processing batch %s-%s: %s", start, end, e)
    return results


async def async_process_batches(
    client: LLMClient,
    message_batches: List[List[Dict[str, str]]],
    *,
    batch_size: int,
    temperature: float,
    parse_fn: Callable[[str], Any],
    raise_on_error: bool = False,
) -> List[Any]:
    """Asynchronously process message batches with :class:`LLMClient`."""

    results: List[Any] = []
    for start in range(0, len(message_batches), batch_size):
        end = min(start + batch_size, len(message_batches))
        batch = message_batches[start:end]
        try:
            responses = await client.async_batch_completion(
                batch,
                temperature=temperature,
                batch_size=batch_size,
            )
            for resp in responses:
                try:
                    results.append(parse_fn(resp))
                except Exception as parse_err:
                    logger.error("Failed to parse response: %s", parse_err)
        except Exception as e:
            if raise_on_error:
                raise
            logger.error("Error processing batch %s-%s: %s", start, end, e)
    return results
