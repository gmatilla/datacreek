# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# DOCX parasers
from typing import Any, Dict

from .base import BaseParser


class DOCXParser(BaseParser):
    """Parser for Microsoft Word documents"""

    def parse(
        self,
        file_path: str,
        *,
        use_unstructured: bool = True,
        return_elements: bool = False
    ) -> str | list[Any]:
        """Parse a DOCX file into plain text using ``unstructured``

        Args:
            file_path: Path to the DOCX file

        Returns:
            Extracted text from the document
        """
        try:
            from unstructured.partition.docx import partition_docx

            elements = partition_docx(filename=file_path)
            if return_elements:
                return elements
            texts = [
                getattr(el, "text", str(el))
                for el in elements
                if getattr(el, "text", None)
            ]
            return "\n".join(texts)
        except Exception as exc:  # pragma: no cover - unexpected failures
            raise RuntimeError("Failed to parse DOCX with unstructured") from exc

    def save(self, content: str, output_path: str) -> None:  # pragma: no cover - legacy
        super().save(content, output_path)
