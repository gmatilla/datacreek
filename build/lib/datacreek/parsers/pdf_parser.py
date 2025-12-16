# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# PDF parser logic
from typing import Any, Dict

from .base import BaseParser


class PDFParser(BaseParser):
    """Parser for PDF documents."""

    def parse(
        self,
        file_path: str,
        *,
        high_res: bool = False,
        ocr: bool = False,
        return_pages: bool = False,
        use_unstructured: bool = True,
        return_elements: bool = False,
    ) -> str | tuple[str, list[str]] | list[Any]:
        """Parse ``file_path`` and return extracted text.

        Parameters
        ----------
        file_path:
            Path to the PDF file.
        high_res:
            If ``True`` and :mod:`llamaparse` is available, use it for
            high resolution parsing of complex layouts.
        ocr:
            When ``True`` additionally run OCR on each page using
            :mod:`pytesseract`.
        """
        text = ""
        pages: list[str] | None = None
        elements: list[Any] | None = None
        if high_res:
            try:
                from llamaparse import LlamaParse

                parser = LlamaParse()
                text = parser.parse(file_path)
            except ImportError as exc:
                raise ImportError(
                    "llamaparse is required for high resolution parsing. Install it with: pip install llamaparse"
                ) from exc
        else:
            try:
                from unstructured.partition.pdf import partition_pdf

                elements = partition_pdf(filename=file_path)
                text = "\n".join(
                    getattr(el, "text", str(el))
                    for el in elements
                    if getattr(el, "text", None)
                )
            except Exception as exc:  # pragma: no cover - unexpected failures
                raise RuntimeError("Failed to parse PDF with unstructured") from exc

            if not text.strip():
                # Fallback OCR if no text was extracted (likely scanned PDF)
                ocr = True

        if ocr:
            try:
                from types import SimpleNamespace

                import pytesseract
                from pdf2image import convert_from_path

                images = convert_from_path(file_path)
                ocr_chunks = [pytesseract.image_to_string(img) for img in images]
                ocr_text = "\n".join(ocr_chunks)
                text += "\n" + ocr_text
                if return_elements:
                    # also expose OCR text as individual elements so downstream
                    # atomisation keeps track of chunk overlap
                    elements = elements or []
                    for chunk in ocr_chunks:
                        elements.append(SimpleNamespace(text=chunk))
            except ImportError as exc:
                raise ImportError(
                    "pdf2image and pytesseract are required for OCR mode. Install them with: pip install pdf2image pytesseract"
                ) from exc

        if return_elements:
            return elements or []

        if return_pages:
            pages = text.split("\f")
            return text, pages

        return text

    def save(self, content: str, output_path: str) -> None:  # pragma: no cover - legacy
        super().save(content, output_path)
