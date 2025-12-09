# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# TXT parsering logic, probably the most minimal
from typing import Any, Dict

from .base import BaseParser


class TXTParser(BaseParser):
    """Parser for plain text files"""

    def parse(self, file_path: str) -> str:
        """Parse a text file

        Args:
            file_path: Path to the text file

        Returns:
            Text content
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def save(self, content: str, output_path: str) -> None:  # pragma: no cover - legacy
        super().save(content, output_path)
