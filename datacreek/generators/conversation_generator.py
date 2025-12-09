import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from datacreek.core.knowledge_graph import KnowledgeGraph
from datacreek.models.llm_client import LLMClient
from datacreek.models.results import ConversationResult
from datacreek.utils import qa_pairs_to_records

from .base import BaseGenerator

logger = logging.getLogger(__name__)


class ConversationGenerator(BaseGenerator):
    """Generate simple conversations from document text.

    This generator builds on :class:`QAGenerator` to first create QA pairs and
    then converts them to a dialogue style dataset. Each conversation keeps track
    of the originating text chunk and source document.
    """

    def __init__(
        self,
        client: LLMClient,
        config_path: Optional[Path] = None,
        kg: Optional[KnowledgeGraph] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(client, config_path, kg=kg, config_overrides=config_overrides)

    def process_document(
        self,
        document_text: str,
        *,
        num_pairs: int = 25,
        verbose: bool = False,
        async_mode: bool = False,
    ) -> ConversationResult:
        """Return conversations generated from ``document_text``."""

        return asyncio.run(
            self._process_document_impl(
                document_text,
                num_pairs=num_pairs,
                verbose=verbose,
                use_async=async_mode,
            )
        )

    async def process_document_async(
        self,
        document_text: str,
        *,
        num_pairs: int = 25,
        verbose: bool = False,
    ) -> ConversationResult:
        """Asynchronous version of :meth:`process_document`."""

        return await self._process_document_impl(
            document_text, num_pairs=num_pairs, verbose=verbose, use_async=True
        )

    async def _process_document_impl(
        self,
        document_text: str,
        *,
        num_pairs: int = 25,
        verbose: bool = False,
        use_async: bool = False,
    ) -> ConversationResult:
        from .qa_generator import QAGenerator

        qa_gen = QAGenerator(
            self.client, self.config_path, kg=self.kg, config_overrides=None
        )

        if use_async:
            result = await qa_gen.process_document_async(
                document_text, num_pairs=num_pairs, verbose=verbose
            )
        else:
            result = await asyncio.to_thread(
                qa_gen.process_document,
                document_text,
                num_pairs=num_pairs,
                verbose=verbose,
            )

        conversations = qa_pairs_to_records(result.qa_pairs)

        return ConversationResult(summary=result.summary, conversations=conversations)
