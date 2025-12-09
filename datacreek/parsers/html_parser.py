# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# HTML Parsers

from typing import Any, Dict
from urllib.parse import urlparse

from .base import BaseParser


class HTMLParser(BaseParser):
    """Parser for HTML files and web pages"""

    def parse(
        self,
        file_path: str,
        *,
        use_unstructured: bool = True,
        return_elements: bool = False
    ) -> str | list[Any]:
        """Parse an HTML file or URL into plain text

        Args:
            file_path: Path to the HTML file or URL

        Returns:
            Extracted text from the HTML
        """
        try:
            from unstructured.partition.html import partition_html

            elements = partition_html(
                url=(
                    file_path if file_path.startswith(("http://", "https://")) else None
                ),
                filename=(
                    None if file_path.startswith(("http://", "https://")) else file_path
                ),
            )
            if return_elements:
                return elements
            texts = [
                getattr(el, "text", str(el))
                for el in elements
                if getattr(el, "text", None)
            ]
            return "\n".join(texts)
        except Exception as exc:  # pragma: no cover - unexpected failures
            raise RuntimeError("Failed to parse HTML with unstructured") from exc

    def save(self, content: str, output_path: str) -> None:  # pragma: no cover - legacy
        super().save(content, output_path)
