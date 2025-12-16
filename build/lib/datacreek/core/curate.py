# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Filter low quality examples

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from datacreek.core.knowledge_graph import KnowledgeGraph
from datacreek.models.llm_client import LLMClient
from datacreek.utils import deduplicate_pairs
from datacreek.utils.config import get_curate_settings, get_prompt


class CurationError(RuntimeError):
    """Raised when QA pair curation encounters parsing errors."""

    def __init__(self, message: str, errors: List[Exception]):
        super().__init__(message)
        self.errors = errors


logger = logging.getLogger(__name__)
from datacreek.models.qa import QAPair
from datacreek.models.results import CurationMetrics, CurationResult
from datacreek.utils.llm_processing import convert_to_conversation_format, parse_ratings


async def _curate_qa_pairs_impl(
    input_data: str | Dict[str, Any],
    threshold: Optional[float],
    api_base: Optional[str],
    model: Optional[str],
    config_path: Optional[Path],
    verbose: bool,
    provider: Optional[str],
    kg: KnowledgeGraph | None,
    batch_size: int | None,
    inference_batch: int | None,
    keep_ratings: bool,
    temperature: float | None,
    resume: bool,
    previous_result: str | Dict[str, Any] | None,
    as_dataclass: bool,
    *,
    use_async_handlers: bool,
) -> Any:
    """Internal coroutine implementing QA pair curation.

    Parameters
    ----------
    as_dataclass:
        Return a :class:`CurationResult` instead of a dictionary when ``True``.
    """

    if isinstance(input_data, str):
        if os.path.exists(input_data):
            data = json.loads(Path(input_data).read_text())
        else:
            data = json.loads(input_data)
    elif isinstance(input_data, dict):
        data = input_data
    else:
        raise TypeError("input_data must be a path, JSON string or dictionary")

    if not isinstance(data, dict):
        raise TypeError("Input data must resolve to a dictionary")

    qa_pairs = data.get("qa_pairs", [])
    summary = data.get("summary", "")
    existing_rated: List[QAPair] = []
    if resume and previous_result is not None:
        try:
            if isinstance(previous_result, str):
                prev = json.loads(previous_result)
            else:
                prev = previous_result
            for p in prev.get("rated_pairs", []):
                existing_rated.append(QAPair(**p))
        except Exception:
            pass
        existing_set = {(p.question, p.answer) for p in existing_rated}
        qa_pairs = [
            p
            for p in qa_pairs
            if (p.get("question"), p.get("answer")) not in existing_set
        ]
    if not isinstance(qa_pairs, list):
        raise ValueError("Input must contain a 'qa_pairs' list")
    if not qa_pairs:
        raise ValueError("No QA pairs found in the input file")

    client = LLMClient(
        config_path=config_path, provider=provider, api_base=api_base, model_name=model
    )

    if threshold is None:
        config = client.config
        threshold = get_curate_settings(config).threshold
    elif not 0 <= threshold <= 10:
        raise ValueError("threshold must be between 0 and 10")

    curate_config = get_curate_settings(client.config)
    batch_size = batch_size or curate_config.batch_size
    inference_batch = inference_batch or curate_config.inference_batch
    rating_temperature = (
        temperature if temperature is not None else curate_config.temperature
    )
    if threshold is None:
        threshold = curate_config.threshold

    if kg:
        try:
            rating_prompt_template = get_prompt(client.config, "kg_qa_rating")
            facts_text = kg.to_text()
        except Exception:
            rating_prompt_template = get_prompt(client.config, "qa_rating")
            facts_text = None
    else:
        rating_prompt_template = get_prompt(client.config, "qa_rating")
        facts_text = None

    batches = [
        qa_pairs[i : i + batch_size] for i in range(0, len(qa_pairs), batch_size)
    ]
    all_messages = []
    for batch in batches:
        batch_json = json.dumps(batch, indent=2)
        if facts_text and "{facts}" in rating_prompt_template:
            rating_prompt = rating_prompt_template.format(
                pairs=batch_json, facts=facts_text
            )
        else:
            rating_prompt = rating_prompt_template.format(pairs=batch_json)
        messages = [{"role": "system", "content": rating_prompt}]
        all_messages.append(messages)

    filtered_pairs = []
    rated_pairs: List[QAPair] = []
    total_score = 0
    total_evaluated = 0
    total_passed = 0
    parse_errors: List[Exception] = []

    logger.info("Processing %d batches of QA pairs...", len(batches))
    if verbose:
        from datacreek.utils.progress import create_progress

        progress_ctx, rate_task = create_progress("Rating QA pairs", len(batches))
        progress_ctx.start()
    else:
        progress_ctx = None
        rate_task = None

    from datacreek.utils.batch import async_process_batches, process_batches

    if use_async_handlers:
        rated_batches = await async_process_batches(
            client,
            all_messages,
            batch_size=inference_batch,
            temperature=rating_temperature,
            parse_fn=lambda resp: resp,
            raise_on_error=True,
        )
    else:
        rated_batches = await asyncio.to_thread(
            process_batches,
            client,
            all_messages,
            batch_size=inference_batch,
            temperature=rating_temperature,
            parse_fn=lambda resp: resp,
            raise_on_error=True,
        )

    def _collect(resp: str, original_batch: List[Dict[str, str]]) -> None:
        rated = parse_ratings(resp, original_batch)
        rated_pairs.extend(rated)
        for pair in rated:
            if pair.rating is not None:
                rating = pair.rating
                if not 0 <= rating <= 10:
                    raise ValueError(f"Rating out of range: {rating}")
                nonlocal total_score, total_evaluated, total_passed
                total_score += rating
                total_evaluated += 1
                if rating >= threshold:
                    filtered_pairs.append(pair.to_dict())
                    total_passed += 1

    for idx, response in enumerate(rated_batches):
        original_batch = batches[idx] if idx < len(batches) else []
        try:
            _collect(response, original_batch)
        except Exception as e:
            parse_errors.append(e)
            logger.error("Error processing batch %d: %s", idx + 1, e)
        if progress_ctx and rate_task:
            progress_ctx.update(rate_task, advance=1)

    if progress_ctx:
        progress_ctx.stop()
    if not verbose:
        logger.info("Batch processing complete.")

    metrics = CurationMetrics(
        total=len(qa_pairs),
        filtered=len(filtered_pairs),
        retention_rate=round(len(filtered_pairs) / len(qa_pairs), 2) if qa_pairs else 0,
        avg_score=round(total_score / total_evaluated, 1) if total_evaluated else 0,
    )

    logger.info("Rated %d QA pairs", total_evaluated)
    logger.info("Retained %d pairs (threshold: %s)", total_passed, threshold)
    logger.info("Average score: %s", metrics.avg_score)

    for p in existing_rated:
        if p.rating is not None and p.rating >= threshold:
            filtered_pairs.append(p.to_dict())
        rated_pairs.append(p)

    before = len(filtered_pairs)
    filtered_pairs = deduplicate_pairs(filtered_pairs)
    if verbose and before != len(filtered_pairs):
        logger.info("Removed %d duplicate pairs", before - len(filtered_pairs))
    if parse_errors:
        raise CurationError("Failed to parse some batches", parse_errors)

    conversations = convert_to_conversation_format(filtered_pairs)

    result = CurationResult(
        summary=summary,
        qa_pairs=[
            QAPair(question=p["question"], answer=p["answer"], rating=p.get("rating"))
            for p in filtered_pairs
        ],
        conversations=conversations,
        metrics=metrics,
        rated_pairs=rated_pairs if keep_ratings else None,
    )

    if as_dataclass:
        return result
    return result.to_dict()


def curate_qa_pairs(
    input_data: str | Dict[str, Any],
    threshold: Optional[float] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    config_path: Optional[Path] = None,
    verbose: bool = False,
    provider: Optional[str] = None,
    async_mode: bool = False,
    kg: KnowledgeGraph | None = None,
    batch_size: int | None = None,
    inference_batch: int | None = None,
    keep_ratings: bool = False,
    temperature: float | None = None,
    resume: bool = False,
    previous_result: str | Dict[str, Any] | None = None,
    as_dataclass: bool = False,
) -> Any:
    """Clean and filter QA pairs based on quality ratings

    Args:
        input_path: Path to the input file with QA pairs
        threshold: Quality threshold (1-10)
        api_base: VLLM API base URL
        model: Model to use
        config_path: Path to configuration file
        verbose: Show detailed output
    async_mode: Use asynchronous LLM requests when supported
    keep_ratings: Return ratings for all pairs even if filtered
    as_dataclass: Return a :class:`CurationResult` instead of a dict

    Returns:
        Cleaned pairs and metrics as a dictionary or dataclass
    """
    return asyncio.run(
        _curate_qa_pairs_impl(
            input_data,
            threshold,
            api_base,
            model,
            config_path,
            verbose,
            provider,
            kg,
            batch_size,
            inference_batch,
            keep_ratings,
            temperature,
            resume,
            previous_result,
            as_dataclass,
            use_async_handlers=async_mode,
        )
    )


async def curate_qa_pairs_async(
    input_data: str | Dict[str, Any],
    threshold: Optional[float] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    config_path: Optional[Path] = None,
    verbose: bool = False,
    provider: Optional[str] = None,
    kg: KnowledgeGraph | None = None,
    batch_size: int | None = None,
    inference_batch: int | None = None,
    keep_ratings: bool = False,
    temperature: float | None = None,
    resume: bool = False,
    previous_result: str | Dict[str, Any] | None = None,
    as_dataclass: bool = False,
) -> Any:
    """Asynchronous version of :func:`curate_qa_pairs`.

    Parameters
    ----------
    as_dataclass:
        Return a :class:`CurationResult` instead of a dictionary when ``True``.
    """

    return await _curate_qa_pairs_impl(
        input_data,
        threshold,
        api_base,
        model,
        config_path,
        verbose,
        provider,
        kg,
        batch_size,
        inference_batch,
        keep_ratings,
        temperature,
        resume,
        previous_result,
        as_dataclass,
        use_async_handlers=True,
    )


def filter_rated_pairs(pairs: List[QAPair], threshold: float) -> List[QAPair]:
    """Return ``pairs`` with rating >= ``threshold``."""

    if not 0 <= threshold <= 10:
        raise ValueError("threshold must be between 0 and 10")
    return [p for p in pairs if p.rating is not None and p.rating >= threshold]


def apply_curation_threshold(
    result: CurationResult | Dict[str, Any], threshold: float
) -> CurationResult:
    """Recompute curated pairs from ``result`` using a new rating ``threshold``."""

    if isinstance(result, dict):
        rated = [QAPair(**p) for p in result.get("rated_pairs", [])]
        base = CurationResult(
            summary=result.get("summary", ""),
            qa_pairs=[],
            conversations=[],
            metrics=CurationMetrics(**result.get("metrics", {})),
            rated_pairs=rated,
        )
    else:
        base = result

    if base.rated_pairs is None:
        raise ValueError("result does not contain rated_pairs")

    filtered = filter_rated_pairs(base.rated_pairs, threshold)
    filtered_pairs = deduplicate_pairs([p.to_dict() for p in filtered])
    conversations = convert_to_conversation_format(filtered_pairs)
    metrics = CurationMetrics(
        total=len(base.rated_pairs),
        filtered=len(filtered),
        retention_rate=(
            round(len(filtered) / len(base.rated_pairs), 2) if base.rated_pairs else 0
        ),
        avg_score=(
            round(
                sum(p.rating for p in base.rated_pairs if p.rating is not None)
                / len(base.rated_pairs),
                1,
            )
            if base.rated_pairs
            else 0
        ),
    )

    return CurationResult(
        summary=base.summary,
        qa_pairs=filtered,
        conversations=conversations,
        metrics=metrics,
        rated_pairs=base.rated_pairs,
    )
