# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Create QA Pairs

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datacreek.core.knowledge_graph import KnowledgeGraph
from datacreek.models.llm_client import LLMClient
from datacreek.models.qa import QAPair
from datacreek.utils.config import (
    get_curate_settings,
    get_generation_config,
    get_prompt,
    load_config,
)
from datacreek.utils.llm_processing import (
    convert_to_conversation_format,
    parse_qa_pairs,
    parse_ratings,
)
from datacreek.utils.progress import create_progress, progress_context
from datacreek.utils.text import split_into_chunks

from .base import BaseGenerator

logger = logging.getLogger(__name__)


class QAGenerator(BaseGenerator):
    def __init__(
        self,
        client: LLMClient,
        config_path: Optional[Path] = None,
        kg: Optional[KnowledgeGraph] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the QA Generator with an LLM client and optional config"""
        super().__init__(client, config_path, kg=kg, config_overrides=config_overrides)
        self.curate_config = get_curate_settings(self.config)

    def generate_summary(
        self, document_text: str, *, verbose: bool | None = None
    ) -> str:
        """Generate a summary of the document"""
        if verbose is None:
            verbose = logger.isEnabledFor(logging.DEBUG)
        if verbose:
            logger.info("Generating document summary...")

        # Get summary prompt from config - prefer KG specific prompt when available
        if self.kg:
            try:
                prompt = get_prompt(self.config, "kg_summary")
                document_text = self.kg.to_text()
            except Exception:
                prompt = get_prompt(self.config, "summary")
        else:
            prompt = get_prompt(self.config, "summary")

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": document_text},
        ]

        temperature = self.generation_config.summary_temperature
        max_tokens = self.generation_config.summary_max_tokens
        summary = self.client.chat_completion(
            messages, temperature=temperature, max_tokens=max_tokens
        )

        if verbose:
            logger.info("Summary generated (%d chars)", len(summary))
        return summary

    def generate_qa_pairs(
        self,
        document_text: str,
        summary: str,
        num_pairs: int = 25,
        query: Optional[str] = None,
        *,
        async_mode: bool = False,
        verbose: bool | None = None,
    ) -> List[QAPair]:
        """Generate QA pairs from the document using batched processing"""

        return asyncio.run(
            self._generate_qa_pairs_impl(
                document_text,
                summary,
                num_pairs=num_pairs,
                query=query,
                verbose=verbose,
                use_async=async_mode,
            )
        )

    async def generate_qa_pairs_async(
        self,
        document_text: str,
        summary: str,
        num_pairs: int = 25,
        query: Optional[str] = None,
        *,
        verbose: bool | None = None,
    ) -> List[QAPair]:  # pragma: no cover - heavy async path
        """Asynchronous counterpart to :meth:`generate_qa_pairs`."""

        return await self._generate_qa_pairs_impl(
            document_text,
            summary,
            num_pairs=num_pairs,
            query=query,
            verbose=verbose,
            use_async=True,
        )

    async def _generate_qa_pairs_impl(  # pragma: no cover - heavy LLM workflow
        self,
        document_text: str,
        summary: str,
        num_pairs: int = 25,
        query: Optional[str] = None,
        *,
        verbose: bool | None = None,
        use_async: bool = False,
    ) -> List[QAPair]:
        if verbose is None:
            verbose = logger.isEnabledFor(logging.DEBUG)

        # Get generation config
        chunk_size = self.generation_config.chunk_size
        temperature = self.generation_config.temperature
        overlap = self.generation_config.overlap
        batch_size = self.generation_config.batch_size
        chunk_method = self.generation_config.chunk_method
        similarity_drop = self.generation_config.similarity_drop
        top_k = self.generation_config.retrieval_top_k

        # Split text into chunks
        chunks = split_into_chunks(
            document_text,
            chunk_size=chunk_size,
            overlap=overlap,
            method=chunk_method,
            similarity_drop=similarity_drop,
        )

        chunk_meta: List[Tuple[str, str, Optional[str]]] = []

        if self.kg and chunk_method in {"sliding", "semantic", "contextual"}:
            # index chunks in knowledge graph if provided
            for i, chunk in enumerate(chunks):
                cid = f"chunk-{i}"
                (
                    self.kg.add_document("doc", source="inline")
                    if "doc" not in self.kg.graph
                    else None
                )
                self.kg.add_chunk("doc", cid, chunk)
                chunk_meta.append((cid, chunk, self.kg.graph.nodes[cid].get("source")))

        if query and self.kg:
            selected_ids = self.kg.search_embeddings(query, k=top_k)
            chunks = [
                self.kg.graph.nodes[c]["text"]
                for c in selected_ids
                if c in self.kg.graph
            ]
            chunk_meta = [
                (
                    cid,
                    self.kg.graph.nodes[cid]["text"],
                    self.kg.graph.nodes[cid].get("source"),
                )
                for cid in selected_ids
                if cid in self.kg.graph
            ]
        else:
            if not chunk_meta:
                chunk_meta = [(f"chunk-{i}", ch, None) for i, ch in enumerate(chunks)]

        if verbose:
            logger.info("Generating QA pairs...")
            logger.info("Document split into %d chunks", len(chunks))
            logger.info("Using batch size of %d", batch_size)

        all_qa_pairs: List[QAPair] = []
        pairs_per_chunk = max(1, round(num_pairs / len(chunks)))

        # Get QA generation prompt template - prefer KG specific one
        if self.kg:
            try:
                qa_prompt_template = get_prompt(self.config, "kg_qa_generation")
            except Exception:
                qa_prompt_template = get_prompt(self.config, "qa_generation")
        else:
            qa_prompt_template = get_prompt(self.config, "qa_generation")

        # Prepare all message batches
        all_messages = []
        for i, chunk in enumerate(chunks):
            # Format the prompt with summary and text
            qa_prompt = qa_prompt_template.format(
                num_pairs=pairs_per_chunk, summary=summary[:100], text=chunk
            )

            messages = [{"role": "system", "content": qa_prompt}]
            all_messages.append(messages)

        logger.info("Processing %d chunks to generate QA pairs...", len(chunks))

        from contextlib import nullcontext

        ctx = (
            progress_context("Generating QA pairs", len(chunks))
            if verbose
            else nullcontext((None, None))
        )
        with ctx as (progress_ctx, generate_task):
            from datacreek.utils.batch import async_process_batches, process_batches

            if use_async:
                batch_results = await async_process_batches(
                    self.client,
                    all_messages,
                    batch_size=batch_size,
                    temperature=temperature,
                    parse_fn=parse_qa_pairs,
                    raise_on_error=True,
                )
            else:
                batch_results = await asyncio.to_thread(
                    process_batches,
                    self.client,
                    all_messages,
                    batch_size=batch_size,
                    temperature=temperature,
                    parse_fn=parse_qa_pairs,
                    raise_on_error=True,
                )

            for i, pairs in enumerate(batch_results):
                cid, chunk_text, src = chunk_meta[i]
                for p in pairs:
                    p.chunk = chunk_text
                    p.source = src or "inline"
                all_qa_pairs.extend(pairs)
                if verbose:
                    logger.info(
                        "  Generated %d pairs from chunk %d",
                        len(pairs),
                        i + 1,
                    )
                if progress_ctx and generate_task:
                    progress_ctx.update(generate_task, advance=1)

        if not verbose:
            logger.info("Batch processing complete.")

        logger.info("Generated %d QA pairs total", len(all_qa_pairs))
        return all_qa_pairs

    def rate_qa_pairs(
        self,
        qa_pairs: List[QAPair],
        summary: str,
        threshold: Optional[float] = None,
        *,
        async_mode: bool = False,
        verbose: bool | None = None,
    ) -> Tuple[List[QAPair], Dict[str, Any]]:
        """Rate and filter QA pairs by quality.

        When ``async_mode`` is ``True`` the LLM calls are executed concurrently
        using :func:`async_process_batches`.
        """
        if verbose is None:
            verbose = logger.isEnabledFor(logging.DEBUG)

        if not qa_pairs:
            return [], {"total": 0, "filtered": 0, "retention_rate": 0, "avg_score": 0}

        # Get threshold from args, then config, then default
        if threshold is None:
            threshold = self.curate_config.threshold

        if verbose:
            logger.info("Evaluating %d pairs...", len(qa_pairs))

        # Get rating config
        batch_size = self.curate_config.batch_size
        temperature = self.curate_config.temperature

        # Get rating prompt template - prefer KG specific one when available
        if self.kg:  # pragma: no cover - heavy
            try:  # pragma: no cover - heavy
                rating_prompt_template = get_prompt(self.config, "kg_qa_rating")
                facts_text = self.kg.to_text()
            except Exception:  # pragma: no cover - heavy
                rating_prompt_template = get_prompt(self.config, "qa_rating")
                facts_text = None
        else:
            rating_prompt_template = get_prompt(self.config, "qa_rating")
            facts_text = None

        # Process in batches
        batches = [
            qa_pairs[i : i + batch_size] for i in range(0, len(qa_pairs), batch_size)
        ]

        rated_pairs: List[QAPair] = []
        total_score = 0.0

        from datacreek.utils.batch import async_process_batches, process_batches

        with progress_context("Rating QA pairs", len(batches)) as (
            progress,
            rating_task,
        ):

            message_batches = []
            for batch in batches:
                batch_dicts = [
                    p.to_dict() if isinstance(p, QAPair) else p for p in batch
                ]
                batch_json = json.dumps(batch_dicts, indent=2)
                if facts_text and "{facts}" in rating_prompt_template:
                    rating_prompt = rating_prompt_template.format(
                        pairs=batch_json, facts=facts_text
                    )
                else:
                    rating_prompt = rating_prompt_template.format(pairs=batch_json)
                message_batches.append([{"role": "system", "content": rating_prompt}])

            if async_mode:  # pragma: no cover - heavy
                responses = asyncio.run(
                    async_process_batches(
                        self.client,
                        message_batches,
                        batch_size=batch_size,
                        temperature=temperature,
                        parse_fn=lambda s: s,
                        raise_on_error=True,
                    )
                )
            else:  # pragma: no cover - heavy
                responses = process_batches(
                    self.client,
                    message_batches,
                    batch_size=batch_size,
                    temperature=temperature,
                    parse_fn=lambda s: s,
                    raise_on_error=True,
                )

            for idx, response in enumerate(responses):
                try:
                    orig_items = [
                        p.to_dict() if isinstance(p, QAPair) else p
                        for p in batches[idx]
                    ]
                    rated_batch = parse_ratings(response, orig_items)
                    for pair in rated_batch:
                        if pair.rating is not None:
                            total_score += pair.rating
                            if pair.rating >= threshold:
                                rated_pairs.append(pair)
                except Exception as e:  # pragma: no cover - heavy error path
                    logger.error("Error processing batch %d: %s", idx + 1, e)

                progress.update(rating_task, advance=1)

        # Calculate metrics
        metrics = {
            "total": len(qa_pairs),
            "filtered": len(rated_pairs),
            "retention_rate": (
                round(len(rated_pairs) / len(qa_pairs), 2) if qa_pairs else 0
            ),
            "avg_score": round(total_score / len(qa_pairs), 1) if qa_pairs else 0,
        }

        # Always print summary information, even in non-verbose mode
        logger.info(
            "Keeping %d out of %d pairs (threshold: %s)",
            len(rated_pairs),
            len(qa_pairs),
            threshold,
        )
        logger.info("Average score: %s", metrics["avg_score"])
        return [p.to_dict() for p in rated_pairs], metrics

    async def rate_qa_pairs_async(  # pragma: no cover - heavy async path
        self,
        qa_pairs: List[QAPair],
        summary: str,
        threshold: Optional[float] = None,
        *,
        verbose: bool | None = None,
    ) -> Tuple[List[QAPair], Dict[str, Any]]:
        """Asynchronous version of :meth:`rate_qa_pairs`."""
        if verbose is None:
            verbose = logger.isEnabledFor(logging.DEBUG)

        if not qa_pairs:
            return [], {"total": 0, "filtered": 0, "retention_rate": 0, "avg_score": 0}

        if threshold is None:
            threshold = self.curate_config.threshold

        if verbose:
            logger.info("Evaluating %d pairs...", len(qa_pairs))

        batch_size = self.curate_config.batch_size
        temperature = self.curate_config.temperature
        if self.kg:
            try:
                rating_prompt_template = get_prompt(self.config, "kg_qa_rating")
                facts_text = self.kg.to_text()
            except Exception:
                rating_prompt_template = get_prompt(self.config, "qa_rating")
                facts_text = None
        else:
            rating_prompt_template = get_prompt(self.config, "qa_rating")
            facts_text = None

        batches = [
            qa_pairs[i : i + batch_size] for i in range(0, len(qa_pairs), batch_size)
        ]
        rated_pairs: List[QAPair] = []
        total_score = 0.0

        from datacreek.utils.batch import async_process_batches

        with progress_context("Rating QA pairs", len(batches)) as (
            progress,
            rating_task,
        ):
            message_batches = []
            for batch in batches:
                batch_dicts = [
                    p.to_dict() if isinstance(p, QAPair) else p for p in batch
                ]
                batch_json = json.dumps(batch_dicts, indent=2)
                if facts_text and "{facts}" in rating_prompt_template:
                    rating_prompt = rating_prompt_template.format(
                        pairs=batch_json, facts=facts_text
                    )
                else:
                    rating_prompt = rating_prompt_template.format(pairs=batch_json)
                message_batches.append([{"role": "system", "content": rating_prompt}])

            responses = await async_process_batches(
                self.client,
                message_batches,
                batch_size=batch_size,
                temperature=temperature,
                parse_fn=lambda s: s,
                raise_on_error=True,
            )

            for idx, response in enumerate(responses):
                try:
                    orig_items = [
                        p.to_dict() if isinstance(p, QAPair) else p
                        for p in batches[idx]
                    ]
                    rated_batch = parse_ratings(response, orig_items)
                    for pair in rated_batch:
                        if pair.rating is not None:
                            total_score += pair.rating
                            if pair.rating >= threshold:
                                rated_pairs.append(pair)
                except Exception as e:
                    logger.error("Error processing batch %d: %s", idx + 1, e)

                progress.update(rating_task, advance=1)

        metrics = {
            "total": len(qa_pairs),
            "filtered": len(rated_pairs),
            "retention_rate": (
                round(len(rated_pairs) / len(qa_pairs), 2) if qa_pairs else 0
            ),
            "avg_score": round(total_score / len(qa_pairs), 1) if qa_pairs else 0,
        }

        logger.info(
            "Keeping %d out of %d pairs (threshold: %s)",
            len(rated_pairs),
            len(qa_pairs),
            threshold,
        )
        logger.info("Average score: %s", metrics["avg_score"])
        return [p.to_dict() for p in rated_pairs], metrics

    def process_document(  # pragma: no cover - heavy entrypoint
        self,
        document_text: str,
        num_pairs: int = 25,
        verbose: bool = False,
        *,
        async_mode: bool = False,
    ) -> "QAGenerationResult":
        """Process a document to generate QA pairs without rating."""

        return asyncio.run(
            self._process_document_impl(
                document_text,
                num_pairs=num_pairs,
                verbose=verbose,
                use_async=async_mode,
            )
        )

    async def process_document_async(  # pragma: no cover - heavy async path
        self,
        document_text: str,
        num_pairs: int = 25,
        verbose: bool = False,
    ) -> "QAGenerationResult":
        """Asynchronous version of :meth:`process_document`."""

        return await self._process_document_impl(
            document_text, num_pairs=num_pairs, verbose=verbose, use_async=True
        )

    async def _process_document_impl(  # pragma: no cover - heavy
        self,
        document_text: str,
        *,
        num_pairs: int = 25,
        verbose: bool = False,
        use_async: bool = False,
    ) -> "QAGenerationResult":
        if use_async:
            summary = await asyncio.to_thread(
                self.generate_summary, document_text, verbose=verbose
            )
        else:
            summary = self.generate_summary(document_text, verbose=verbose)

        qa_pairs = await self._generate_qa_pairs_impl(
            document_text,
            summary,
            num_pairs=num_pairs,
            verbose=verbose,
            use_async=use_async,
        )
        from datacreek.models.results import QAGenerationResult

        return QAGenerationResult(summary=summary, qa_pairs=qa_pairs)
