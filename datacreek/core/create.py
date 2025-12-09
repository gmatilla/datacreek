# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Generate the content: CoT/QA/Summary Datasets
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import redis
except Exception:  # pragma: no cover - optional dependency missing

    class _RedisStub:
        Redis = object

    redis = _RedisStub()  # type: ignore

from datacreek.core.knowledge_graph import KnowledgeGraph
from datacreek.models.content_type import ContentType
from datacreek.models.llm_client import LLMClient
from datacreek.storage import StorageBackend
from datacreek.utils.config import get_generation_config, load_config

logger = logging.getLogger(__name__)


def load_document_text(file_path: str) -> str:
    """Return the raw text from ``file_path``."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def init_llm_client(
    config_path: Optional[Path],
    *,
    provider: Optional[str] = None,
    profile: Optional[str] = None,
    api_base: Optional[str] = None,
    model_name: Optional[str] = None,
) -> LLMClient:
    """Instantiate :class:`LLMClient` with common parameters."""
    return LLMClient(
        config_path=config_path,
        provider=provider,
        profile=profile,
        api_base=api_base,
        model_name=model_name,
    )


def _base_name(file_path: Optional[str]) -> str:
    return os.path.splitext(os.path.basename(file_path))[0] if file_path else "input"


def process_file(
    file_path: Optional[str],
    config_path: Optional[Path] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    content_type: ContentType | str = ContentType.QA,
    num_pairs: Optional[int] = None,
    verbose: bool = False,
    *,
    async_mode: bool = False,
    provider: Optional[str] = None,
    profile: Optional[str] = None,
    kg: KnowledgeGraph,
    document_text: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    multi_answer: bool = False,
    redis_client: "redis.Redis" | None = None,
    redis_key: str | None = None,
    backend: "StorageBackend" | None = None,
) -> Any:
    """Process a file to generate content

    Args:
        file_path: Path to the text file to process
        config_path: Path to configuration file
        api_base: VLLM API base URL
        model: Model to use
        content_type: Type of content to generate. Can be a
            :class:`~datacreek.models.content_type.ContentType` value or string
            ("qa", "summary", "cot", "cot-enhance", "vqa_add_reasoning").
        num_pairs: Target number of QA pairs to generate
        async_mode: Use asynchronous LLM requests when supported
        kg: Knowledge graph built during ingestion
        multi_answer: Generate multiple answers per fact when using
            the knowledge graph generator
        redis_client: Optional Redis connection to store the result
        redis_key: Key used when persisting to Redis
        backend: Optional storage backend overriding ``redis_client``

    Returns:
        The generated result or the backend key when persistence is used
    """
    client = init_llm_client(
        config_path,
        provider=provider,
        profile=profile,
        api_base=api_base,
        model_name=model,
    )

    logger.info("Using %s provider", client.provider)

    ct = ContentType(content_type)

    def _generate_qa() -> Any:  # pragma: no cover - heavy generation logic
        from datacreek.generators.qa_generator import QAGenerator

        generator = QAGenerator(
            client, config_path, kg=kg, config_overrides=config_overrides
        )

        nonlocal document_text
        if document_text is None:
            document_text = kg.to_text()

        if num_pairs is None:
            config = client.config
            generation_config = get_generation_config(config)
            num_pairs_local = generation_config.num_pairs
        else:
            num_pairs_local = num_pairs

        if async_mode:
            gen_result = asyncio.run(
                generator.process_document_async(
                    document_text,
                    num_pairs=num_pairs_local,
                    verbose=verbose,
                )
            )
        else:
            gen_result = generator.process_document(
                document_text,
                num_pairs=num_pairs_local,
                verbose=verbose,
            )

        return gen_result.to_dict()

    def _generate_summary() -> Any:  # pragma: no cover - summarization helper
        generator = QAGenerator(
            client, config_path, kg=kg, config_overrides=config_overrides
        )

        nonlocal document_text
        if document_text is None:
            document_text = kg.to_text()

        summary = generator.generate_summary(document_text, verbose=verbose)

        return {"summary": summary}

    def _generate_cot() -> Any:  # pragma: no cover - complex CoT logic
        from datacreek.generators.cot_generator import COTGenerator

        generator = COTGenerator(client, config_path, config_overrides=config_overrides)

        nonlocal document_text
        if document_text is None:
            document_text = kg.to_text()

        if num_pairs is None:
            config = client.config
            generation_config = get_generation_config(config)
            num_examples = generation_config.num_cot_examples
        else:
            num_examples = num_pairs

        cot_result = generator.process_document(
            document_text,
            num_examples=num_examples,
            include_simple_steps=verbose,
        )

        return cot_result.to_dict()

    def _cot_enhance() -> Any:  # pragma: no cover - heavy JSON handling
        from tqdm import tqdm

        from datacreek.generators.cot_generator import COTGenerator

        # Initialize the CoT generator
        generator = COTGenerator(client, config_path, config_overrides=config_overrides)

        if document_text is None:
            document_text = kg.to_text()

        # Get max_examples from args or config
        max_examples = None
        if num_pairs is not None:
            max_examples = num_pairs  # If user specified a number, use it
        else:
            config = client.config
            generation_config = get_generation_config(config)
            # Get the config value (will be None by default, meaning enhance all)
            max_examples = generation_config.num_cot_enhance_examples

        # Instead of parsing as text, load the file as JSON with conversations
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(
                f"Failed to parse {file_path} as JSON. For cot-enhance, input must be a valid JSON file."
            )

        # Handle different dataset formats
        # First, check for QA pairs format (the most common input format)
        if isinstance(data, dict) and "qa_pairs" in data:
            from datacreek.utils.llm_processing import convert_to_conversation_format

            qa_pairs = data.get("qa_pairs", [])
            if verbose:
                logger.debug(
                    "Converting %d QA pairs to conversation format", len(qa_pairs)
                )

            conv_list = convert_to_conversation_format(qa_pairs)
            conversations = [{"conversations": conv} for conv in conv_list]
            is_single_conversation = False
        elif isinstance(data, dict) and "conversations" in data:
            conversations = [data]
            is_single_conversation = True
        elif isinstance(data, list) and all(
            "conversations" in item for item in data if isinstance(item, dict)
        ):
            conversations = data
            is_single_conversation = False
        elif isinstance(data, list) and all(
            isinstance(msg, dict) and "from" in msg for msg in data
        ):
            conversations = [{"conversations": data}]
            is_single_conversation = True
        else:
            conversations = data
            is_single_conversation = False

            # Limit the number of conversations if needed
            if max_examples is not None and len(conversations) > max_examples:
                if verbose:
                    logger.debug(
                        "Limiting to %d conversations (from %d total)",
                        max_examples,
                        len(conversations),
                    )
                conversations = conversations[:max_examples]

            if verbose:
                logger.debug("Found %d conversation(s) to enhance", len(conversations))

            # Process each conversation
            enhanced_conversations = []

            for i, conversation in enumerate(
                tqdm(conversations, desc="Enhancing conversations")
            ):
                # Check if this item has a conversations field
                if isinstance(conversation, dict) and "conversations" in conversation:
                    conv_messages = conversation["conversations"]

                    # Validate messages format
                    if not isinstance(conv_messages, list):
                        logger.warning(
                            "conversations field is not a list in item %d, skipping", i
                        )
                        enhanced_conversations.append(conversation)  # Keep original
                        continue

                    # Enhance this conversation's messages
                    if verbose:
                        logger.debug("Conv_messages type: %s", type(conv_messages))
                        logger.debug(
                            "Conv_messages structure: %s",
                            (
                                conv_messages[:1]
                                if isinstance(conv_messages, list)
                                else "Not a list"
                            ),
                        )

                    # Always include simple steps when enhancing QA pairs
                    enhanced_messages = generator.enhance_with_cot(
                        conv_messages, include_simple_steps=True
                    )

                    # Handle nested bug
                    if enhanced_messages and isinstance(enhanced_messages, list):
                        # Nested bug
                        if enhanced_messages and isinstance(enhanced_messages[0], list):
                            if verbose:
                                logger.debug("Flattening nested array response")
                            enhanced_messages = enhanced_messages[0]

                    # Create enhanced conversation with same structure
                    enhanced_conv = conversation.copy()
                    enhanced_conv["conversations"] = enhanced_messages
                    enhanced_conversations.append(enhanced_conv)
                else:
                    # Not the expected format, just keep original
                    enhanced_conversations.append(conversation)

        if is_single_conversation and len(enhanced_conversations) == 1:
            return enhanced_conversations[0]
        return enhanced_conversations

    def _vqa_reasoning() -> Any:  # pragma: no cover - heavy dataset logic
        from datacreek.generators.vqa_generator import VQAGenerator

        generator = VQAGenerator(client, config_path, config_overrides=config_overrides)

        return generator.process_dataset(
            dataset_source=file_path,
            num_examples=num_pairs,
            verbose=verbose,
        )

    def _generate_from_kg() -> Any:  # pragma: no cover - complex graph ops
        from datacreek.generators.kg_generator import KGGenerator

        nonlocal document_text
        if document_text is None:
            document_text = kg.to_text()

        generator = KGGenerator(client, config_path, config_overrides=config_overrides)

        result = generator.process_graph(
            kg,
            num_pairs=num_pairs or get_generation_config(client.config).num_pairs,
            verbose=verbose,
            multi_answer=multi_answer,
        )

        return result

    def _tool_call() -> Any:  # pragma: no cover - I/O heavy
        from datacreek.generators.tool_generator import ToolCallGenerator

        nonlocal document_text
        if document_text is None:
            document_text = kg.to_text()

        generator = ToolCallGenerator(
            client, config_path, kg=kg, config_overrides=config_overrides
        )
        result = generator.process_document(document_text, verbose=verbose)
        return result

    def _conversation() -> Any:  # pragma: no cover - heavy
        from datacreek.generators.conversation_generator import ConversationGenerator

        nonlocal document_text
        if document_text is None:
            document_text = kg.to_text()

        generator = ConversationGenerator(
            client, config_path, kg=kg, config_overrides=config_overrides
        )
        result = generator.process_document(document_text, verbose=verbose)
        return result

    def _multi_tool() -> Any:  # pragma: no cover - heavy
        from datacreek.generators.multi_tool_generator import MultiToolGenerator

        nonlocal document_text
        if document_text is None:
            document_text = kg.to_text()

        generator = MultiToolGenerator(
            client, config_path, kg=kg, config_overrides=config_overrides
        )
        result = generator.process_document(document_text, verbose=verbose)
        return result

    def _pref_pair() -> Any:  # pragma: no cover - heavy
        from datacreek.generators.pref_generator import PrefPairGenerator

        nonlocal document_text
        if document_text is None:
            document_text = kg.to_text()

        generator = PrefPairGenerator(
            client, config_path, kg=kg, config_overrides=config_overrides
        )
        result = generator.process_document(document_text, verbose=verbose)
        return result

    def _pref_list() -> Any:  # pragma: no cover - heavy
        from datacreek.generators.pref_generator import PrefListGenerator

        nonlocal document_text
        if document_text is None:
            document_text = kg.to_text()

        generator = PrefListGenerator(
            client, config_path, kg=kg, config_overrides=config_overrides
        )
        result = generator.process_document(document_text, verbose=verbose)
        return result

    handlers = {
        ContentType.QA: _generate_qa,
        ContentType.SUMMARY: _generate_summary,
        ContentType.COT: _generate_cot,
        ContentType.COT_ENHANCE: _cot_enhance,
        ContentType.VQA_ADD_REASONING: _vqa_reasoning,
        ContentType.FROM_KG: _generate_from_kg,
        ContentType.TOOL_CALL: _tool_call,
        ContentType.CONVERSATION: _conversation,
        ContentType.MULTI_TOOL: _multi_tool,
        ContentType.PREF_PAIR: _pref_pair,
        ContentType.PREF_LIST: _pref_list,
    }

    if ct not in handlers:  # pragma: no cover - sanity check
        raise ValueError(f"Unknown content type: {content_type}")

    result = handlers[ct]()
    if backend is not None and redis_key:  # pragma: no cover - I/O wrapper
        try:
            return backend.save(redis_key, json.dumps(result))
        except Exception:  # pragma: no cover - error path
            logger.exception("Failed to save generated data via backend")
            raise
    if redis_client and redis_key:  # pragma: no cover - I/O wrapper
        try:
            redis_client.set(redis_key, json.dumps(result))
            return redis_key
        except Exception:  # pragma: no cover - error path
            logger.exception("Failed to save generated data to Redis")
            raise
    return result


async def process_file_async(  # pragma: no cover - thin wrapper
    file_path: Optional[str],
    config_path: Optional[Path] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    content_type: ContentType | str = ContentType.QA,
    num_pairs: Optional[int] = None,
    verbose: bool = False,
    *,
    provider: Optional[str] = None,
    profile: Optional[str] = None,
    kg: KnowledgeGraph,
    document_text: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    multi_answer: bool = False,
) -> Any:
    """Asynchronous version of :func:`process_file`."""

    if ContentType(content_type) is not ContentType.QA:
        return await asyncio.to_thread(
            process_file,
            file_path,
            config_path,
            api_base,
            model,
            content_type,
            num_pairs,
            verbose,
            async_mode=False,
            provider=provider,
            profile=profile,
            kg=kg,
            document_text=document_text,
            config_overrides=config_overrides,
            multi_answer=multi_answer,
        )

    client = init_llm_client(
        config_path,
        provider=provider,
        profile=profile,
        api_base=api_base,
        model_name=model,
    )

    if document_text is None:
        document_text = kg.to_text()

    from datacreek.generators.qa_generator import QAGenerator

    generator = QAGenerator(
        client, config_path, kg=kg, config_overrides=config_overrides
    )
    if num_pairs is None:
        num_pairs = get_generation_config(client.config).num_pairs

    result = await generator.process_document_async(
        document_text,
        num_pairs=num_pairs,
        verbose=verbose,
    )

    return result.to_dict()
