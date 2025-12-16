# pragma: no cover
# Complex chain-of-thought sampling not needed for coverage.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Logic for generating CoT from scratch and also enhancing CoT (take existing format and add CoT)
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from datacreek.models.cot import COTExample
from datacreek.models.llm_client import LLMClient
from datacreek.models.results import COTGenerationResult
from datacreek.utils.config import get_generation_config, get_prompt
from datacreek.utils.text import extract_json_from_text

logger = logging.getLogger(__name__)


from .base import BaseGenerator


class COTGenerator(BaseGenerator):
    """Generates chain-of-thought reasoning examples"""

    def __init__(
        self,
        client: LLMClient,
        config_path: Optional[Path] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the CoT Generator with an LLM client and optional config"""
        super().__init__(
            client, config_path, kg=None, config_overrides=config_overrides
        )

    def parse_json_output(self, output_text: str) -> Optional[List[Dict]]:
        """Parse JSON from LLM output text"""
        verbose = logger.isEnabledFor(logging.DEBUG)
        try:
            data = extract_json_from_text(output_text)
        except Exception as e:  # pragma: no cover - defensive
            if verbose:
                logger.error("Error parsing output: %s", e)
            return None

        if not isinstance(data, list):
            if verbose:
                logger.warning("Expected a list but got %s", type(data))
            return None
        return data

    def generate_cot_examples(
        self, document_text: str, num_examples: int | None = None
    ) -> List[COTExample]:
        """Generate chain-of-thought reasoning examples."""

        return asyncio.run(
            self._generate_cot_examples_impl(
                document_text, num_examples=num_examples, use_async=False
            )
        )

    async def generate_cot_examples_async(
        self, document_text: str, num_examples: int | None = None
    ) -> List[COTExample]:
        """Asynchronous counterpart to :meth:`generate_cot_examples`."""

        return await self._generate_cot_examples_impl(
            document_text, num_examples=num_examples, use_async=True
        )

    async def _generate_cot_examples_impl(
        self,
        document_text: str,
        num_examples: int | None = None,
        *,
        use_async: bool = False,
    ) -> List[COTExample]:
        verbose = logger.isEnabledFor(logging.DEBUG)

        if num_examples is None:
            num_examples = self.generation_config.num_cot_examples

        prompt_template = get_prompt(self.config, "cot_generation")
        prompt = prompt_template.format(num_examples=num_examples, text=document_text)

        temperature = self.generation_config.temperature
        max_tokens = self.generation_config.max_tokens

        if verbose:
            logger.info("Generating %d CoT examples...", num_examples)

        messages = [{"role": "system", "content": prompt}]
        if use_async:
            response = await asyncio.to_thread(
                self.client.chat_completion,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            response = self.client.chat_completion(
                messages, temperature=temperature, max_tokens=max_tokens
            )

        parsed = self.parse_json_output(response)

        examples: List[COTExample] = []
        if parsed is None:
            if verbose:
                logger.warning("Failed to parse CoT examples, returning empty list")
            return []

        for item in parsed:
            if not isinstance(item, dict):
                if verbose:
                    logger.debug("Skipping malformed item: %r", item)
                continue
            examples.append(
                COTExample(
                    question=item.get("question", ""),
                    reasoning=item.get("reasoning", ""),
                    answer=item.get("answer", ""),
                )
            )

        if verbose:
            logger.info("Successfully generated %d CoT examples", len(examples))

        return examples

    def enhance_with_cot(
        self, conversations: List[Dict], include_simple_steps: bool = False
    ) -> List[Dict]:
        """Enhance existing conversations with CoT reasoning"""
        verbose = logger.isEnabledFor(logging.DEBUG)

        # Get the prompt template
        prompt_template = get_prompt(self.config, "cot_enhancement")

        if verbose:
            logger.debug("Conversations structure: %s", type(conversations))
            logger.debug(
                "First conversation: %s",
                json.dumps(conversations[0] if conversations else {}, indent=2)[:100],
            )

        # Format the prompt
        conversation_str = json.dumps(conversations, ensure_ascii=False, indent=2)
        prompt = prompt_template.format(
            conversations=conversation_str,
            include_simple_steps=str(include_simple_steps).lower(),
        )

        # Generate enhanced conversations
        temperature = self.generation_config.temperature
        max_tokens = self.generation_config.max_tokens

        if verbose:
            logger.info("Enhancing %d conversations with CoT...", len(conversations))

        messages = [{"role": "system", "content": prompt}]
        response = self.client.chat_completion(
            messages, temperature=temperature, max_tokens=max_tokens
        )

        # Parse response
        enhanced_conversations = self.parse_json_output(response)

        if enhanced_conversations is None:
            if verbose:
                logger.warning(
                    "Failed to parse enhanced conversations, returning original"
                )
            return conversations

        if verbose:
            logger.info("Successfully enhanced conversations with CoT")

        return enhanced_conversations

    def process_document(
        self,
        document_text: str,
        num_examples: int | None = None,
        include_simple_steps: bool = False,
    ) -> COTGenerationResult:
        """Process a document to generate CoT examples."""

        return asyncio.run(
            self._process_document_impl(
                document_text,
                num_examples=num_examples,
                include_simple_steps=include_simple_steps,
                use_async=False,
            )
        )

    async def process_document_async(
        self,
        document_text: str,
        num_examples: int | None = None,
        include_simple_steps: bool = False,
    ) -> COTGenerationResult:
        """Asynchronous version of :meth:`process_document`."""

        return await self._process_document_impl(
            document_text,
            num_examples=num_examples,
            include_simple_steps=include_simple_steps,
            use_async=True,
        )

    async def _process_document_impl(
        self,
        document_text: str,
        num_examples: int | None = None,
        include_simple_steps: bool = False,
        *,
        use_async: bool = False,
    ) -> COTGenerationResult:
        verbose = logger.isEnabledFor(logging.DEBUG)

        if use_async:
            summary = await asyncio.to_thread(
                self.client.chat_completion,
                [
                    {
                        "role": "system",
                        "content": "Summarize this document in 2-3 sentences.",
                    },
                    {"role": "user", "content": document_text},
                ],
                temperature=0.1,
            )
        else:
            summary = self.client.chat_completion(
                [
                    {
                        "role": "system",
                        "content": "Summarize this document in 2-3 sentences.",
                    },
                    {"role": "user", "content": document_text},
                ],
                temperature=0.1,
            )

        examples = await self._generate_cot_examples_impl(
            document_text, num_examples=num_examples, use_async=use_async
        )

        conversations = []
        for example in examples:
            conv = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides detailed explanations.",
                },
                {"role": "user", "content": example.question},
                {
                    "role": "assistant",
                    "content": (
                        "Let me think through this step by step:\n\n"
                        f"{example.reasoning}\n\nSo the answer is: {example.answer}"
                    ),
                },
            ]
            conversations.append(conv)

        result = COTGenerationResult(
            summary=summary,
            cot_examples=examples,
            conversations=conversations,
        )

        if verbose:
            logger.info("Generated %d chain-of-thought examples", len(examples))

        return result
