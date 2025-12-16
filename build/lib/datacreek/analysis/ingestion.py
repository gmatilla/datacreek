"""Lightweight ingestion utilities."""

from __future__ import annotations

from typing import List


def partition_files_to_atoms(path: str) -> List[str]:
    """Return textual atoms from ``path`` using unstructured if available."""
    try:
        from unstructured.partition.auto import partition

        elements = partition(path)
        texts = [getattr(el, "text", "") for el in elements]
        return [t.strip() for t in texts if t and t.strip()]
    except Exception:  # pragma: no cover - optional dependency may be missing
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = f.read()
        return [t.strip() for t in data.splitlines() if t.strip()]


def transcribe_audio(path: str) -> str:
    """Return transcription of ``path`` using Whisper if available."""
    try:  # pragma: no cover - heavy optional dependency
        import whisper

        model = whisper.load_model("tiny")
        result = model.transcribe(path)
        return result.get("text", "").strip()
    except Exception:
        return ""


def blip_caption_image(path: str) -> str:
    """Return image caption using BLIP if available."""
    try:  # pragma: no cover - heavy optional dependency
        from PIL import Image
        from transformers import BlipForConditionalGeneration, BlipProcessor

        img = Image.open(path)
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        inputs = processor(img, return_tensors="pt")
        out = model.generate(**inputs)
        return processor.decode(out[0], skip_special_tokens=True)
    except Exception:
        return ""


def parse_code_to_atoms(path: str) -> List[str]:
    """Return code atoms (functions/classes) from a Python file.

    Parameters
    ----------
    path: str
        Path to the ``.py`` file to parse.

    Returns
    -------
    List[str]
        List of code snippets representing each top-level function or class.
    """
    try:
        import ast
        import textwrap

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            source = textwrap.dedent(f.read())
        tree = ast.parse(source)
        atoms: List[str] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                segment = ast.get_source_segment(source, node)
                if segment:
                    atoms.append(segment.strip())
        return atoms if atoms else [source.strip()]
    except Exception:  # pragma: no cover
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return [f.read().strip()]
