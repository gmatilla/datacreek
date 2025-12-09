import sys
import types

import pytest

pytest.importorskip("PIL")
from PIL import Image

sys.modules.setdefault("torch", types.ModuleType("torch"))

import datacreek.analysis.monitoring
import datacreek.parsers
from datacreek import DatasetBuilder, DatasetType, ingest_file, to_kg
from datacreek.core.ingest import ingest_into_dataset, process_file
from datacreek.parsers import ImageParser, WhisperAudioParser
from datacreek.parsers.pdf_parser import PDFParser
from datacreek.utils.config import load_config
from datacreek.utils.modality import detect_modality


def test_ingest_to_kg(tmp_path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("Hello world. This is a test document.")

    ds = DatasetBuilder(DatasetType.TEXT)
    text = ingest_file(str(text_file))
    to_kg(text, ds, "doc1")

    assert ds.search_chunks("Hello")
    assert ds.get_chunks_for_document("doc1")


def test_to_kg_no_index_build(tmp_path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("Hello world")

    ds = DatasetBuilder(DatasetType.TEXT)
    text = ingest_file(str(text_file))
    to_kg(text, ds, "doc1", build_index=False)

    assert ds.graph.index._vectorizer is None
    # calling search should auto-build the index
    assert ds.search_chunks("Hello") == ["doc1_chunk_0"]


def test_chunk_overlap_metadata(tmp_path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("Hello world")

    cfg = load_config()
    cfg.setdefault("ingest", {})["chunk_overlap"] = 2
    cfg["ingest"]["chunk_size"] = 5

    ds = DatasetBuilder(DatasetType.TEXT)
    text = ingest_file(str(text_file))
    to_kg(text, ds, "d1", config=cfg)
    cid = ds.get_chunks_for_document("d1")[0]
    assert ds.graph.graph.nodes[cid]["overlap"] == 2


def test_to_kg_with_pages(tmp_path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("Page1\fPage2")

    ds = DatasetBuilder(DatasetType.TEXT)
    to_kg("Page1\fPage2", ds, "doc1", pages=["Page1", "Page2"])

    assert ds.get_page_for_chunk("doc1_chunk_0") == 1
    assert ds.get_page_for_chunk("doc1_chunk_1") == 2


def test_to_kg_with_elements(tmp_path, monkeypatch):
    class El:
        def __init__(self, text=None, image_path=None, page_number=1):
            self.text = text
            self.image_path = image_path
            self.metadata = types.SimpleNamespace(
                page_number=page_number, image_path=image_path
            )

    elements = [
        El("Hello", page_number=1),
        El(image_path="img.png", page_number=1),
        El("World", page_number=1),
    ]

    ds = DatasetBuilder(DatasetType.TEXT)
    with monkeypatch.context() as m:
        m.setitem(
            sys.modules,
            "datacreek.utils.image_captioning",
            types.SimpleNamespace(
                caption_images_parallel=lambda paths, **kw: ["cap" for _ in paths]
            ),
        )
        to_kg("Hello\nWorld", ds, "doc1", elements=elements)

    assert ds.get_images_for_document("doc1") == ["doc1_image_0"]
    assert ds.graph.graph.nodes["doc1_image_0"].get("alt_text") == "cap"
    assert len(ds.get_chunks_for_document("doc1")) == 2
    assert ds.get_atoms_for_document("doc1") == ["doc1_atom_0", "doc1_atom_1"]
    assert ds.get_molecules_for_document("doc1") == [
        "doc1_molecule_0",
        "doc1_molecule_1",
    ]
    from datacreek.utils.modality import detect_modality

    assert ds.graph.graph.nodes["doc1_atom_0"].get("modality") == detect_modality(
        "Hello"
    )
    chunk_id = ds.get_chunks_for_document("doc1")[0]
    assert ds.graph.graph.nodes[chunk_id].get("modality") == detect_modality("Hello")


def test_image_deduplication(monkeypatch, tmp_path):
    class El:
        def __init__(self, image_path=None, page_number=1):
            self.image_path = image_path
            self.metadata = types.SimpleNamespace(
                page_number=page_number, image_path=image_path
            )

    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.png"
    Image.new("RGB", (8, 8), color="red").save(img1)
    Image.new("RGB", (8, 8), color="red").save(img2)

    elements = [El(image_path=str(img1)), El(image_path=str(img2))]

    ds = DatasetBuilder(DatasetType.TEXT)
    ds._enforce_policy = lambda *a, **k: None
    called = {"n": 0}

    def fake_caption(paths, **kw):
        called["n"] += len(paths)
        return [f"cap-{p}" for p in paths]

    with monkeypatch.context() as m:
        m.setitem(
            sys.modules,
            "datacreek.utils.image_captioning",
            types.SimpleNamespace(caption_images_parallel=fake_caption),
        )
        from datacreek.utils import image_dedup

        image_dedup.FILTER[:] = b"\x00" * len(image_dedup.FILTER)
        to_kg("img", ds, "docx", elements=elements)

    imgs = ds.get_images_for_document("docx")
    assert len(imgs) == 2
    for img in imgs:
        node = ds.graph.graph.nodes[img]
        if node["path"] == str(img1):
            assert node.get("alt_text") == f"cap-{img1}"
        else:
            assert node.get("alt_text") is None
    assert called["n"] == 1


def test_ingest_audio(tmp_path, monkeypatch):
    audio_file = tmp_path / "s.wav"
    audio_file.write_text("fake")

    class DummyParser(datacreek.parsers.WhisperAudioParser):
        def parse(self, file_path: str) -> str:
            return "hello world"

    monkeypatch.setattr(
        datacreek.core.ingest, "determine_parser", lambda f, c: DummyParser()
    )
    ds = DatasetBuilder(DatasetType.TEXT)
    ingest_into_dataset(str(audio_file), ds, doc_id="a1")

    assert ds.get_audios_for_document("a1") == ["a1_audio_0"]
    cid = ds.get_chunks_for_document("a1")[0]
    assert (cid, "a1_audio_0") in ds.graph.graph.edges


def test_to_kg_extract_entities(tmp_path, monkeypatch):
    class El:
        def __init__(self, text=None, page_number=1):
            self.text = text
            self.metadata = types.SimpleNamespace(page_number=page_number)

    elements = [El("Paris is nice", page_number=1)]

    ds = DatasetBuilder(DatasetType.TEXT)
    with monkeypatch.context() as m:
        m.setitem(
            sys.modules,
            "datacreek.utils.image_captioning",
            types.SimpleNamespace(caption_image=lambda p: "cap"),
        )
        to_kg("Paris is nice", ds, "doc2", elements=elements, extract_entities=True)

    chunk_id = ds.get_chunks_for_document("doc2")[0]
    assert "Paris" in ds.graph.graph.nodes[chunk_id].get("entities", [])


def test_determine_parser_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest_file(str(tmp_path / "missing.txt"))
    bad_file = tmp_path / "file.badext"
    bad_file.write_text("x")
    with pytest.raises(ValueError):
        ingest_file(str(bad_file))


def test_ingest_into_dataset_with_facts(tmp_path):
    text_file = tmp_path / "doc.txt"
    text_file.write_text("Paris is the capital of France.")
    ds = DatasetBuilder(DatasetType.TEXT)
    ingest_into_dataset(str(text_file), ds, extract_facts=True)
    facts = ds.search_facts("Paris")
    assert facts


def test_ingest_into_dataset_with_entities(tmp_path):
    text_file = tmp_path / "doc.txt"
    text_file.write_text("Albert Einstein was born in Ulm.")
    ds = DatasetBuilder(DatasetType.TEXT)
    ingest_into_dataset(str(text_file), ds, extract_entities=True)
    ents = ds.search_entities("Einstein")
    assert ents


def test_ingest_into_dataset_records_source(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("hello")

    ds = DatasetBuilder(DatasetType.TEXT)
    ingest_into_dataset(str(f), ds)

    assert ds.graph.graph.nodes["sample"]["source"] == str(f)


def test_ingest_image_and_audio(monkeypatch, tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"x")
    aud = tmp_path / "clip.wav"
    aud.write_bytes(b"y")

    monkeypatch.setattr(ImageParser, "parse", lambda self, p: "img text")
    monkeypatch.setattr(ImageParser, "save", lambda self, c, o: None)
    monkeypatch.setattr(WhisperAudioParser, "parse", lambda self, p: "audio text")
    monkeypatch.setattr(WhisperAudioParser, "save", lambda self, c, o: None)

    ds = DatasetBuilder(DatasetType.TEXT)
    ingest_into_dataset(str(img), ds)
    ingest_into_dataset(str(aud), ds)

    assert ds.search_chunks("img text")
    assert ds.search_chunks("audio text")


def test_determine_parser_remote_image(monkeypatch):
    called = {}

    class DummyParser:
        def parse(self, file_path):
            called["path"] = file_path
            return "ok"

    monkeypatch.setitem(datacreek.parsers._PARSER_REGISTRY, ".png", DummyParser)

    text = ingest_file("https://example.com/test.png")
    assert text == "ok"
    assert called["path"] == "https://example.com/test.png"


def test_determine_parser_remote_audio(monkeypatch):
    called = {}

    class DummyParser:
        def parse(self, file_path):
            called["path"] = file_path
            return "ok"

    monkeypatch.setitem(datacreek.parsers._PARSER_REGISTRY, ".mp3", DummyParser)

    text = ingest_file("https://example.com/test.mp3")
    assert text == "ok"
    assert called["path"] == "https://example.com/test.mp3"


def test_process_file_unstructured(monkeypatch, tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"x")
    called = {}

    class Dummy(PDFParser):
        def parse(
            self,
            file_path,
            *,
            high_res=False,
            ocr=False,
            return_pages=False,
            use_unstructured=False,
            return_elements=False,
        ):
            assert use_unstructured is True
            return "ok"

    monkeypatch.setattr(datacreek.core.ingest, "determine_parser", lambda f, c: Dummy())
    text = process_file(str(pdf), use_unstructured=True)
    assert text == "ok"


def test_ingest_logs_metrics(tmp_path, caplog, monkeypatch):
    f = tmp_path / "doc.txt"
    f.write_text("hello world")

    ds = DatasetBuilder(DatasetType.TEXT)
    calls = []
    monkeypatch.setattr(
        datacreek.analysis.monitoring,
        "update_metric",
        lambda n, v: calls.append((n, v)),
    )
    with caplog.at_level("INFO"):
        ingest_into_dataset(str(f), ds)

    assert ds.graph.graph.graph.get("n_atoms") == 0
    assert ds.graph.graph.graph.get("avg_chunk_len", 0) > 0
    assert any("n_atoms=" in rec.message for rec in caplog.records)
    assert ("atoms_total", 0.0) in calls
