# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Ingest different file formats

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:  # optional dependency
    from pydantic import BaseModel
except Exception:  # pragma: no cover - optional dependency missing

    class BaseModel:  # lightweight stub for missing pydantic
        pass


from datacreek.analysis.monitoring import blip_called_total, blip_skipped_total
from datacreek.core.dataset import DatasetBuilder
from datacreek.models.llm_client import LLMClient
from datacreek.parsers import (
    HTMLParser,
    PDFParser,
    YouTubeParser,
    get_parser_for_extension,
)
from datacreek.utils.config import get_generation_config, load_config
from datacreek.utils.image_dedup import check_duplicate
from datacreek.utils.image_quality import should_caption
from datacreek.utils.modality import detect_modality
from datacreek.utils.text import clean_text, split_into_chunks

logger = logging.getLogger(__name__)

# Directory containing user uploads. When set, ingestion is restricted to this
# path to avoid reading arbitrary files from the host machine.
UPLOAD_ROOT = os.getenv("DATACREEK_UPLOAD_DIR")


def validate_file_path(path: str) -> None:
    """Ensure ``path`` resides inside :data:`UPLOAD_ROOT` when configured."""

    if path.startswith(("http://", "https://")):
        return
    if not UPLOAD_ROOT:
        return
    abs_path = os.path.abspath(path)
    root = os.path.abspath(UPLOAD_ROOT)
    if not abs_path.startswith(root):
        raise ValueError(f"Access to {path} is not allowed")


@dataclass
class IngestOptions:
    """Configuration for ingesting a document."""

    config: Optional[Dict[str, Any]] = None
    high_res: bool = False
    ocr: bool = False
    use_unstructured: bool | None = None
    extract_entities: bool = False
    extract_facts: bool = False
    compute_metrics: bool = False
    emotion_fn: Callable[[str], str] | None = None
    modality_fn: Callable[[str], str] | None = None


class IngestOptionsModel(BaseModel):
    """Pydantic validation model for :class:`IngestOptions`."""

    config: Optional[Dict[str, Any]] = None
    high_res: bool = False
    ocr: bool = False
    use_unstructured: bool | None = None
    extract_entities: bool = False
    extract_facts: bool = False

    def to_options(self) -> IngestOptions:
        """Convert to :class:`IngestOptions`."""
        return IngestOptions(**self.model_dump())


def determine_parser(file_path: str, config: Dict[str, Any]):
    """Return a parser instance for the given resource."""
    # Check if it's a URL
    if file_path.startswith(("http://", "https://")):
        if "youtube.com" in file_path or "youtu.be" in file_path:
            return YouTubeParser()
        from urllib.parse import urlparse

        path = urlparse(file_path).path
        ext = os.path.splitext(path)[1].lower()
        parser = get_parser_for_extension(ext)
        if parser:
            return parser
        return HTMLParser()

    if not os.path.exists(file_path):
        logger.error("File not found: %s", file_path)
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    parser = get_parser_for_extension(ext)
    if parser:
        return parser
    logger.error("Unsupported file extension: %s", ext)
    raise ValueError(f"Unsupported file extension: {ext}")


def process_file(
    file_path: str,
    config: Optional[Dict[str, Any]] = None,
    *,
    high_res: bool = False,
    ocr: bool = False,
    return_pages: bool = False,
    use_unstructured: bool | None = None,
    return_elements: bool = False,
) -> str | list[Any] | tuple[str, list[str]]:
    """Parse ``file_path`` and return the extracted text."""

    cfg = config or load_config()
    if use_unstructured is None:
        use_unstructured = cfg.get("ingest", {}).get("use_unstructured", True)

    parser = determine_parser(file_path, cfg)
    parse_kwargs = {}
    if isinstance(parser, PDFParser):
        parse_kwargs = {"high_res": high_res, "ocr": ocr, "return_pages": return_pages}
    if hasattr(parser.parse, "__call__"):
        if "use_unstructured" in parser.parse.__code__.co_varnames:
            parse_kwargs["use_unstructured"] = use_unstructured
        if return_elements and "return_elements" in parser.parse.__code__.co_varnames:
            parse_kwargs["return_elements"] = True
    content = parser.parse(file_path, **parse_kwargs)

    # Fallback OCR with pytesseract if unstructured extraction returned nothing
    if isinstance(parser, PDFParser) and (
        (isinstance(content, str) and not content.strip())
        or (isinstance(content, list) and not content)
    ):
        try:
            from types import SimpleNamespace

            import pytesseract
            from pdf2image import convert_from_path

            lang = cfg.get("ingest", {}).get("ocr_lang", "eng")
            images = convert_from_path(file_path)
            ocr_texts = [pytesseract.image_to_string(img, lang=lang) for img in images]
            if return_elements:
                content = [SimpleNamespace(text=t) for t in ocr_texts]
            else:
                content = "\n".join(ocr_texts)
        except Exception:  # pragma: no cover - optional deps may be missing
            pass

    if return_pages and isinstance(content, tuple):
        text, pages = content
    elif return_elements and isinstance(content, list):
        elements = content
        text = "\n".join(
            getattr(el, "text", str(el)) for el in elements if getattr(el, "text", None)
        )
        pages = None
    else:
        text = content
        pages = None

    if return_pages:
        return text, pages or []
    return text


def to_kg(
    text: str,
    dataset: DatasetBuilder,
    doc_id: str,
    config: Optional[Dict[str, Any]] = None,
    *,
    build_index: bool = True,
    pages: list[str] | None = None,
    elements: list[Any] | None = None,
    source: str | None = None,
    checksum: str | None = None,
    extract_entities: bool = False,
    progress_callback: Callable[[int], None] | None = None,
    emotion_fn: Callable[[str], str] | None = None,
    modality_fn: Callable[[str], str] | None = None,
) -> None:
    """Split ``text`` and populate ``dataset`` with nodes.

    Parameters
    ----------
    build_index: bool, optional
        Whether to rebuild the embedding index after inserting chunks. Set to
        ``False`` when ingesting many files sequentially to build the index once
        at the end.
    """

    cfg = config or load_config()
    gen_cfg = get_generation_config(cfg)
    ingest_cfg = cfg.get("ingest", {})
    chunk_size = ingest_cfg.get("chunk_size", gen_cfg.chunk_size)
    chunk_overlap = ingest_cfg.get("chunk_overlap", gen_cfg.overlap)

    cleaned_text = clean_text(text)
    cleaned_pages = [clean_text(p) for p in pages] if pages else None

    uid = hashlib.sha1(
        cleaned_text.encode("utf-8"), usedforsecurity=False
    ).hexdigest()  # nosec B324

    dataset.add_document(
        doc_id,
        source=source or doc_id,
        text=cleaned_text,
        checksum=checksum,
        uid=uid,
    )

    if elements:
        chunk_idx = 0
        atom_idx = 0
        molecule_idx = 0
        atoms: list[str] = []
        img_idx = 0
        current_page = 1
        pending_imgs: list[tuple[str, str, int]] = []
        for el in elements:
            page = (
                getattr(getattr(el, "metadata", None), "page_number", None)
                or current_page
            )
            if page != current_page and atoms:
                mol_id = f"{doc_id}_molecule_{molecule_idx}"
                dataset.add_molecule(doc_id, mol_id, atoms)
                molecule_idx += 1
                atoms = []
            current_page = page
            path = getattr(el, "image_path", None) or getattr(
                getattr(el, "metadata", None), "image_path", None
            )
            if path:
                if atoms:
                    mol_id = f"{doc_id}_molecule_{molecule_idx}"
                    dataset.add_molecule(doc_id, mol_id, atoms)
                    molecule_idx += 1
                    atoms = []
                img_id = f"{doc_id}_image_{img_idx}"
                pending_imgs.append((img_id, path, page))
                img_idx += 1
                if len(pending_imgs) >= 256:
                    try:
                        from datacreek.utils.image_captioning import (
                            caption_images_parallel,
                        )
                    except Exception:  # pragma: no cover - optional dep failure
                        caption_images_parallel = None

                    to_caption: list[tuple[str, str, int]] = []
                    for item in pending_imgs:
                        pid, ppath, ppage = item
                        dup = False
                        try:
                            dup = check_duplicate(ppath)
                        except Exception:
                            dup = False
                        quality = True
                        try:
                            quality = should_caption(ppath)
                        except Exception:
                            quality = True
                        if dup or not quality:
                            if blip_skipped_total is not None:
                                blip_skipped_total.inc()
                            dataset.add_image(
                                doc_id, pid, ppath, page=ppage, alt_text=""
                            )
                        else:
                            if blip_called_total is not None:
                                blip_called_total.inc()
                            to_caption.append(item)

                    alts: list[str] = ["" for _ in to_caption]
                    if to_caption and caption_images_parallel is not None:
                        try:
                            alts = caption_images_parallel([p[1] for p in to_caption])
                        except Exception:  # pragma: no cover - optional dep failure
                            alts = ["" for _ in to_caption]
                    for (pid, ppath, ppage), alt in zip(to_caption, alts):
                        dataset.add_image(doc_id, pid, ppath, page=ppage, alt_text=alt)
                    pending_imgs = []
                continue
            text_el = getattr(el, "text", None)
            if not text_el:
                continue
            cleaned_el = clean_text(text_el)
            atom_id = f"{doc_id}_atom_{atom_idx}"
            if emotion_fn is None:
                try:
                    from datacreek.utils.emotion import detect_emotion as _emo
                except Exception:
                    _emo = None
                emotion_fn = _emo
            atom_emotion = emotion_fn(cleaned_el) if emotion_fn else None
            if modality_fn is None:
                try:
                    from datacreek.utils.modality import detect_modality as _mod
                except Exception:
                    _mod = None
                modality_fn = _mod
            atom_modality = modality_fn(cleaned_el) if modality_fn else None
            atom_entities = None
            if extract_entities:
                try:
                    from datacreek.utils.entity_extraction import (
                        extract_entities as extract_entities_fn,
                    )

                    atom_entities = extract_entities_fn(cleaned_el, model=None)
                except Exception:
                    atom_entities = None
            dataset.add_atom(
                doc_id,
                atom_id,
                cleaned_el,
                el.__class__.__name__,
                page=page,
                emotion=atom_emotion,
                modality=atom_modality,
                entities=atom_entities,
            )
            atoms.append(atom_id)
            atom_idx += 1
            chunks = split_into_chunks(
                cleaned_el,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
                method=gen_cfg.chunk_method,
                similarity_drop=gen_cfg.similarity_drop,
            )
            for chunk in chunks:
                cid = f"{doc_id}_chunk_{chunk_idx}"
                if emotion_fn is None:
                    try:
                        from datacreek.utils.emotion import detect_emotion as _em
                    except Exception:
                        _em = None
                    emotion_fn = _em
                emotion = emotion_fn(chunk) if emotion_fn else None
                if modality_fn is None:
                    try:
                        from datacreek.utils.modality import detect_modality as _mod
                    except Exception:
                        _mod = None
                    modality_fn = _mod
                modality = modality_fn(chunk) if modality_fn else None
                entities = None
                if extract_entities:
                    try:
                        from datacreek.utils.entity_extraction import (
                            extract_entities as extract_entities_fn,
                        )

                        entities = extract_entities_fn(chunk, model=None)
                    except Exception:
                        entities = None
                dataset.add_chunk(
                    doc_id,
                    cid,
                    chunk,
                    page=page,
                    emotion=emotion,
                    modality=modality,
                    entities=entities,
                    chunk_overlap=chunk_overlap,
                )
                chunk_idx += 1
                if progress_callback:
                    progress_callback(chunk_idx)
        if pending_imgs:
            try:
                from datacreek.utils.image_captioning import caption_images_parallel
            except Exception:  # pragma: no cover - optional dep failure
                caption_images_parallel = None

            to_caption: list[tuple[str, str, int]] = []
            for pid, ppath, ppage in pending_imgs:
                dup = False
                try:
                    dup = check_duplicate(ppath)
                except Exception:
                    dup = False
                quality = True
                try:
                    quality = should_caption(ppath)
                except Exception:
                    quality = True
                if dup or not quality:
                    if blip_skipped_total is not None:
                        blip_skipped_total.inc()
                    dataset.add_image(doc_id, pid, ppath, page=ppage, alt_text="")
                else:
                    if blip_called_total is not None:
                        blip_called_total.inc()
                    to_caption.append((pid, ppath, ppage))

            alts: list[str] = ["" for _ in to_caption]
            if to_caption and caption_images_parallel is not None:
                try:
                    alts = caption_images_parallel([p[1] for p in to_caption])
                except Exception:  # pragma: no cover - optional dep failure
                    alts = ["" for _ in to_caption]
            for (pid, ppath, ppage), alt in zip(to_caption, alts):
                dataset.add_image(doc_id, pid, ppath, page=ppage, alt_text=alt)
        if atoms:
            mol_id = f"{doc_id}_molecule_{molecule_idx}"
            dataset.add_molecule(doc_id, mol_id, atoms)
    elif cleaned_pages:
        chunk_idx = 0
        for page_num, page_text in enumerate(cleaned_pages, start=1):
            page_chunks = split_into_chunks(
                page_text,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
                method=gen_cfg.chunk_method,
                similarity_drop=gen_cfg.similarity_drop,
            )
            for chunk in page_chunks:
                cid = f"{doc_id}_chunk_{chunk_idx}"
                if emotion_fn is None:
                    try:
                        from datacreek.utils.emotion import detect_emotion as _em
                    except Exception:
                        _em = None
                    emotion_fn = _em
                emotion = emotion_fn(chunk) if emotion_fn else None
                if modality_fn is None:
                    try:
                        from datacreek.utils.modality import detect_modality as _mod
                    except Exception:
                        _mod = None
                    modality_fn = _mod
                modality = modality_fn(chunk) if modality_fn else None
                entities = None
                if extract_entities:
                    try:
                        from datacreek.utils.entity_extraction import (
                            extract_entities as extract_entities_fn,
                        )

                        entities = extract_entities_fn(chunk, model=None)
                    except Exception:
                        entities = None
                dataset.add_chunk(
                    doc_id,
                    cid,
                    chunk,
                    page=page_num,
                    emotion=emotion,
                    modality=modality,
                    entities=entities,
                    chunk_overlap=chunk_overlap,
                )
                chunk_idx += 1
                if progress_callback:
                    progress_callback(chunk_idx)
    else:
        chunks = split_into_chunks(
            cleaned_text,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            method=gen_cfg.chunk_method,
            similarity_drop=gen_cfg.similarity_drop,
        )
        for i, chunk in enumerate(chunks):
            cid = f"{doc_id}_chunk_{i}"
            if emotion_fn is None:
                try:
                    from datacreek.utils.emotion import detect_emotion as _em
                except Exception:
                    _em = None
                emotion_fn = _em
            emotion = emotion_fn(chunk) if emotion_fn else None
            try:
                modality = detect_modality(chunk)
            except Exception:
                modality = None
            entities = None
            if extract_entities:
                try:
                    from datacreek.utils.entity_extraction import (
                        extract_entities as extract_entities_fn,
                    )

                    entities = extract_entities_fn(chunk, model=None)
                except Exception:
                    entities = None
            dataset.add_chunk(
                doc_id,
                cid,
                chunk,
                emotion=emotion,
                modality=modality,
                entities=entities,
                chunk_overlap=chunk_overlap,
            )
            if progress_callback:
                progress_callback(i + 1)

    n_atoms = len(dataset.get_atoms_for_document(doc_id))
    chunks = dataset.get_chunks_for_document(doc_id)
    total_len = sum(len(dataset.graph.graph.nodes[c].get("text", "")) for c in chunks)
    avg_len = total_len / len(chunks) if chunks else 0.0
    dataset.graph.graph.graph["n_atoms"] = n_atoms
    dataset.graph.graph.graph["avg_chunk_len"] = avg_len
    logger.info("n_atoms=%d avg_chunk_len=%.2f", n_atoms, avg_len)
    try:
        from datacreek.analysis.monitoring import update_metric

        update_metric("atoms_total", float(n_atoms))
        update_metric("avg_chunk_len", float(avg_len))
    except Exception:  # pragma: no cover - optional Prometheus
        pass

    if build_index:
        dataset.graph.index.build()


def ingest_into_dataset(
    file_path: str,
    dataset: DatasetBuilder,
    doc_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    *,
    high_res: bool = False,
    ocr: bool = False,
    use_unstructured: bool | None = None,
    extract_entities: bool = False,
    extract_facts: bool = False,
    client: "LLMClient" | None = None,
    options: IngestOptions | None = None,
    progress_callback: Callable[[int], None] | None = None,
    compute_metrics: bool = False,
    emotion_fn: Callable[[str], str] | None = None,
    modality_fn: Callable[[str], str] | None = None,
) -> str:
    """Parse ``file_path`` and populate ``dataset`` with its content.

    Parameters
    ----------
    options:
        Optional :class:`IngestOptions` overriding individual parameters.
    extract_entities:
        Run NER over inserted chunks if ``True``.
    extract_facts:
        Extract simple subject/predicate/object facts from chunks if ``True``.
        When enabled, ``client`` may supply an :class:`~datacreek.models.llm_client.LLMClient`
        instance to use for extraction.
    """

    if options is not None:
        config = options.config
        high_res = options.high_res
        ocr = options.ocr
        use_unstructured = options.use_unstructured
        extract_entities = options.extract_entities
        extract_facts = options.extract_facts
        compute_metrics = options.compute_metrics
        emotion_fn = options.emotion_fn
        modality_fn = options.modality_fn

    validate_file_path(file_path)

    try:
        result = process_file(
            file_path,
            config=config,
            high_res=high_res,
            ocr=ocr,
            use_unstructured=use_unstructured,
            return_pages=True,
            return_elements=True,
        )
    except Exception:
        logger.exception("Failed to parse %s", file_path)
        raise
    elements = None
    if isinstance(result, list):
        elements = result
        text = "\n".join(
            getattr(el, "text", str(el)) for el in elements if getattr(el, "text", None)
        )
        pages = None
    elif isinstance(result, tuple):
        text, pages = result
    else:
        text = result
        pages = None
    doc_id = doc_id or Path(file_path).stem
    checksum = None
    try:
        from datacreek.utils.checksum import md5_file

        checksum = md5_file(file_path)
    except Exception:
        pass
    try:
        to_kg(
            text,
            dataset,
            doc_id,
            config,
            build_index=True,
            pages=pages,
            elements=elements,
            source=file_path,
            checksum=checksum,
            extract_entities=extract_entities,
            progress_callback=progress_callback,
            emotion_fn=emotion_fn,
            modality_fn=modality_fn,
        )
    except Exception:
        logger.exception("Failed to build knowledge graph for %s", file_path)
        raise

    try:
        parser = determine_parser(file_path, config or load_config())
        from datacreek.parsers import WhisperAudioParser, YouTubeParser

        if isinstance(parser, (WhisperAudioParser, YouTubeParser)):
            audio_id = f"{doc_id}_audio_0"
            try:
                from datacreek.utils.text import detect_language

                lang = detect_language(text)
            except Exception:
                lang = "und"
            dataset.add_audio(doc_id, audio_id, file_path, lang=lang)
            for cid in dataset.graph.get_chunks_for_document(doc_id):
                dataset.graph.link_transcript(cid, audio_id, provenance=file_path)
    except Exception:  # pragma: no cover - optional deps may be missing
        pass

    if extract_entities:
        try:
            dataset.extract_entities()
        except Exception:
            logger.exception("Failed to extract entities from %s", file_path)
            raise
    if extract_facts:
        try:
            dataset.extract_facts(client)
        except Exception:
            logger.exception("Failed to extract facts from %s", file_path)
            raise
        dataset.history.append("Facts extracted on ingest")
    if compute_metrics:
        try:
            dataset.fractal_information_metrics([1])
        except Exception:
            logger.exception("Failed to compute fractal metrics for %s", file_path)
    return doc_id


async def ingest_into_dataset_async(
    file_path: str,
    dataset: DatasetBuilder,
    doc_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    *,
    high_res: bool = False,
    ocr: bool = False,
    use_unstructured: bool | None = None,
    extract_entities: bool = False,
    extract_facts: bool = False,
    client: "LLMClient" | None = None,
    options: IngestOptions | None = None,
    progress_callback: Callable[[int], None] | None = None,
    compute_metrics: bool = False,
    emotion_fn: Callable[[str], str] | None = None,
    modality_fn: Callable[[str], str] | None = None,
) -> str:
    """Asynchronous wrapper around :func:`ingest_into_dataset`."""

    return await asyncio.to_thread(
        ingest_into_dataset,
        file_path,
        dataset,
        doc_id,
        config,
        high_res=high_res,
        ocr=ocr,
        use_unstructured=use_unstructured,
        extract_entities=extract_entities,
        extract_facts=extract_facts,
        client=client,
        options=options,
        progress_callback=progress_callback,
        compute_metrics=compute_metrics,
        emotion_fn=emotion_fn,
        modality_fn=modality_fn,
    )
