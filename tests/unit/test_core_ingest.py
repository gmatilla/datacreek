import asyncio
import sys
import types
from pathlib import Path

import pytest

# Stub heavy optional dependencies so the ingest module imports without them.
sys.modules.setdefault(
    "imagehash",
    types.SimpleNamespace(
        phash=lambda img: types.SimpleNamespace(
            hash=types.SimpleNamespace(tobytes=lambda: b"0" * 8)
        )
    ),
)
dummy_img_mod = types.SimpleNamespace(
    open=lambda path: types.SimpleNamespace(close=lambda: None)
)
sys.modules.setdefault("PIL", types.SimpleNamespace(Image=dummy_img_mod))
sys.modules.setdefault("PIL.Image", dummy_img_mod)

import datacreek.core.ingest as ing
from datacreek.core.dataset_light import DatasetBuilder
from datacreek.pipelines import DatasetType


def test_validate_file_path_restricted(monkeypatch, tmp_path):
    monkeypatch.setattr(ing, "UPLOAD_ROOT", str(tmp_path))
    with pytest.raises(ValueError):
        ing.validate_file_path("/outside.txt")


def test_determine_parser_url(monkeypatch):
    class Dummy:
        pass

    monkeypatch.setattr(ing, "YouTubeParser", lambda: Dummy())
    parser = ing.determine_parser("https://youtube.com/watch?v=1", {})
    assert isinstance(parser, Dummy)


def test_determine_parser_unknown(monkeypatch, tmp_path):
    monkeypatch.setattr(ing, "get_parser_for_extension", lambda ext: None)
    p = tmp_path / "file.bad"
    p.write_text("x")
    with pytest.raises(ValueError):
        ing.determine_parser(str(p), {})


def test_ingest_into_dataset(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr(
        ing, "validate_file_path", lambda p: calls.setdefault("val", True)
    )
    monkeypatch.setattr(ing, "process_file", lambda *a, **k: "hello")
    monkeypatch.setattr(ing, "determine_parser", lambda *a, **k: object())
    captured = {}

    def fake_to_kg(text, dataset, doc_id, config=None, **kw):
        captured["text"] = text
        captured["doc_id"] = doc_id

    monkeypatch.setattr(ing, "to_kg", fake_to_kg)

    dataset = DatasetBuilder(DatasetType.TEXT)
    f = tmp_path / "doc.txt"
    f.write_text("content")
    result = ing.ingest_into_dataset(str(f), dataset)
    assert result == "doc"
    assert captured["text"] == "hello"
    assert captured["doc_id"] == "doc"


@pytest.mark.asyncio
async def test_ingest_into_dataset_async(monkeypatch):
    monkeypatch.setattr(ing, "ingest_into_dataset", lambda *a, **k: "done")
    result = await ing.ingest_into_dataset_async("f", DatasetBuilder(DatasetType.TEXT))
    assert result == "done"


import types


class DummyGraph:
    def __init__(self):
        class G:
            pass

        self.graph = G()
        self.graph.nodes = {}
        self.graph.graph = {}
        import types

        self.index = types.SimpleNamespace(build=lambda: None)

    def add_chunk(self, doc_id, chunk_id, text):
        self.graph.nodes[chunk_id] = {"text": text}

    def get_chunks_for_document(self, doc_id):
        return [cid for cid in self.graph.nodes if cid.startswith(f"{doc_id}_")]

    def get_atoms_for_document(self, doc_id):
        return []

    def link_transcript(self, cid, audio_id, provenance=None):
        pass


class DummyDataset(DatasetBuilder):
    def __init__(self):
        super().__init__(DatasetType.TEXT)
        self.graph = DummyGraph()
        self.add_molecule_calls = []
        self.add_image_calls = []
        self.add_atom_calls = []
        self.add_chunk_calls = []
        self.add_audio_calls = []

    def add_document(self, doc_id, source, text=None, checksum=None, uid=None):
        self.doc = doc_id

    def add_molecule(self, doc_id, mol_id, atoms):
        self.add_molecule_calls.append((mol_id, atoms))

    def add_image(self, doc_id, img_id, path, page=None, alt_text=""):
        self.add_image_calls.append((img_id, path, page, alt_text))

    def add_atom(self, doc_id, atom_id, text, element_type, **kwargs):
        self.add_atom_calls.append((atom_id, text))

    def add_chunk(self, doc_id, chunk_id, text, **kwargs):
        self.add_chunk_calls.append((chunk_id, text))
        self.graph.add_chunk(doc_id, chunk_id, text)

    def add_audio(self, doc_id, audio_id, path, lang=None):
        self.add_audio_calls.append((audio_id, path, lang))

    def get_atoms_for_document(self, doc_id):
        return self.graph.get_atoms_for_document(doc_id)

    def get_chunks_for_document(self, doc_id):
        return self.graph.get_chunks_for_document(doc_id)


def make_cfg():
    class Cfg:
        chunk_size = 2
        overlap = 0
        chunk_method = "default"
        similarity_drop = 0.0

    return Cfg


def setup_ingest(monkeypatch):
    monkeypatch.setattr(
        ing, "load_config", lambda: {"ingest": {"chunk_size": 2, "chunk_overlap": 0}}
    )
    monkeypatch.setattr(ing, "get_generation_config", lambda cfg: make_cfg())
    monkeypatch.setattr(ing, "clean_text", lambda t: t)
    monkeypatch.setattr(ing, "split_into_chunks", lambda text, **kw: [text])


def test_to_kg_pages(monkeypatch):
    setup_ingest(monkeypatch)
    ds = DummyDataset()
    ing.to_kg("abc", ds, "d1", {}, pages=["p1", "p2"])
    assert len(ds.add_chunk_calls) == 2


def test_to_kg_elements(monkeypatch):
    setup_ingest(monkeypatch)
    monkeypatch.setattr(ing, "check_duplicate", lambda path: False)
    monkeypatch.setattr(ing, "should_caption", lambda path: True)
    ds = DummyDataset()
    el = types.SimpleNamespace(
        text="img", image_path="img.png", metadata=types.SimpleNamespace(page_number=1)
    )
    ing.to_kg("text", ds, "d2", {}, elements=[el])
    assert ds.add_image_calls


def test_to_kg_image_batch(monkeypatch):
    setup_ingest(monkeypatch)
    monkeypatch.setattr(ing, "check_duplicate", lambda p: False)
    monkeypatch.setattr(ing, "should_caption", lambda p: True)
    monkeypatch.setitem(
        sys.modules,
        "datacreek.utils.image_captioning",
        types.SimpleNamespace(
            caption_images_parallel=lambda paths: [
                f"alt{i}" for i, _ in enumerate(paths)
            ]
        ),
    )
    ds = DummyDataset()
    elements = [
        types.SimpleNamespace(
            text="t", image_path="p.png", metadata=types.SimpleNamespace(page_number=1)
        )
        for _ in range(260)
    ]
    ing.to_kg(
        "text",
        ds,
        "d3",
        {},
        elements=elements,
        emotion_fn=lambda t: "e",
        modality_fn=lambda t: "m",
        extract_entities=True,
    )
    assert len(ds.add_image_calls) == 260


def test_to_kg_default(monkeypatch):
    setup_ingest(monkeypatch)
    monkeypatch.setattr(
        ing, "detect_modality", lambda t: (_ for _ in ()).throw(ValueError())
    )
    ds = DummyDataset()
    ing.to_kg(
        "hello",
        ds,
        "d4",
        {},
        extract_entities=True,
        emotion_fn=lambda t: "e",
        modality_fn=lambda t: "m",
        progress_callback=lambda n: None,
    )
    assert ds.add_chunk_calls


def test_to_kg_text_elements(monkeypatch):
    setup_ingest(monkeypatch)
    monkeypatch.setattr(ing, "check_duplicate", lambda p: False)
    monkeypatch.setattr(ing, "should_caption", lambda p: True)
    monkeypatch.setitem(
        sys.modules,
        "datacreek.utils.entity_extraction",
        types.SimpleNamespace(extract_entities=lambda text, model=None: ["ent"]),
    )
    ds = DummyDataset()
    elem = types.SimpleNamespace(
        text="hello", metadata=types.SimpleNamespace(page_number=2)
    )
    progress = []
    ing.to_kg(
        "text",
        ds,
        "d5",
        {},
        elements=[elem],
        extract_entities=True,
        progress_callback=lambda n: progress.append(n),
    )
    assert ds.add_atom_calls and ds.add_chunk_calls and progress


def test_to_kg_pages_full(monkeypatch):
    setup_ingest(monkeypatch)
    monkeypatch.setattr(ing, "detect_modality", lambda t: "m")
    monkeypatch.setitem(
        sys.modules,
        "datacreek.utils.emotion",
        types.SimpleNamespace(detect_emotion=lambda t: "e"),
    )
    monkeypatch.setitem(
        sys.modules,
        "datacreek.utils.entity_extraction",
        types.SimpleNamespace(
            extract_entities=lambda text, model=None: (_ for _ in ()).throw(
                RuntimeError()
            )
        ),
    )
    ds = DummyDataset()
    progress = []
    ing.to_kg(
        "p1\np2",
        ds,
        "d6",
        {},
        pages=["p1", "p2"],
        extract_entities=True,
        progress_callback=lambda n: progress.append(n),
    )
    assert progress and ds.add_chunk_calls


def test_to_kg_no_options(monkeypatch):
    setup_ingest(monkeypatch)
    monkeypatch.setitem(
        sys.modules,
        "datacreek.utils.emotion",
        types.SimpleNamespace(detect_emotion=lambda t: "e"),
    )
    ds = DummyDataset()
    ing.to_kg("abcd", ds, "d7", {})
    assert ds.add_chunk_calls


def test_ingest_parse_error(monkeypatch, tmp_path):
    dataset = DummyDataset()
    monkeypatch.setattr(ing, "validate_file_path", lambda p: None)
    monkeypatch.setattr(
        ing, "process_file", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail"))
    )
    f = tmp_path / "x.txt"
    f.write_text("x")
    with pytest.raises(RuntimeError):
        ing.ingest_into_dataset(str(f), dataset)


def test_ingest_checksum_error(monkeypatch, tmp_path):
    dataset = DummyDataset()
    monkeypatch.setattr(ing, "validate_file_path", lambda p: None)
    monkeypatch.setattr(ing, "process_file", lambda *a, **k: "txt")
    monkeypatch.setattr(ing, "determine_parser", lambda *a, **k: object())
    monkeypatch.setattr(ing, "to_kg", lambda *a, **k: None)
    monkeypatch.setitem(
        sys.modules,
        "datacreek.utils.checksum",
        types.SimpleNamespace(md5_file=lambda p: (_ for _ in ()).throw(RuntimeError())),
    )
    f = tmp_path / "y.txt"
    f.write_text("x")
    ing.ingest_into_dataset(str(f), dataset)


def test_ingest_into_dataset_branches(monkeypatch, tmp_path):
    class CaptureDataset(DummyDataset):
        def __init__(self):
            super().__init__()
            self.entities = False
            self.facts = False
            self.metrics = False
            self.history = []

        def extract_entities(self):
            self.entities = True

        def extract_facts(self, client):
            self.facts = True

        def fractal_information_metrics(self, arr):
            self.metrics = True

    dataset = CaptureDataset()
    monkeypatch.setattr(ing, "validate_file_path", lambda p: None)
    monkeypatch.setattr(ing, "process_file", lambda *a, **k: "txt")
    monkeypatch.setattr(ing, "determine_parser", lambda *a, **k: object())
    monkeypatch.setattr(ing, "to_kg", lambda *a, **k: None)
    f = tmp_path / "d.txt"
    f.write_text("x")
    ing.ingest_into_dataset(
        str(f),
        dataset,
        extract_entities=True,
        extract_facts=True,
        compute_metrics=True,
        client=object(),
    )
    assert dataset.entities and dataset.facts and dataset.metrics


def test_ingest_audio_transcript(monkeypatch, tmp_path):
    dataset = DummyDataset()
    dataset.graph.link_calls = []

    def link(cid, aid, provenance=None):
        dataset.graph.link_calls.append((cid, aid))

    dataset.graph.link_transcript = link
    dataset.graph.get_chunks_for_document = lambda doc_id: ["d_audio_chunk"]

    class DummyParser:
        pass

    import datacreek.parsers as parsers

    monkeypatch.setattr(parsers, "WhisperAudioParser", DummyParser)
    monkeypatch.setattr(parsers, "YouTubeParser", DummyParser)
    monkeypatch.setattr(ing, "determine_parser", lambda *a, **k: DummyParser())
    monkeypatch.setattr(ing, "process_file", lambda *a, **k: "text")
    monkeypatch.setattr(ing, "to_kg", lambda *a, **k: None)

    f = tmp_path / "audio.wav"
    f.write_text("x")
    ing.ingest_into_dataset(str(f), dataset)
    assert dataset.add_audio_calls and dataset.graph.link_calls


import sys


def test_process_file_kwargs(monkeypatch):
    class DummyParser:
        def parse(self, file_path, use_unstructured=None, return_elements=False):
            DummyParser.called = (use_unstructured, return_elements)
            return [types.SimpleNamespace(text="a"), types.SimpleNamespace(text="b")]

    monkeypatch.setattr(ing, "determine_parser", lambda *a, **k: DummyParser())
    monkeypatch.setattr(
        ing, "load_config", lambda: {"ingest": {"use_unstructured": True}}
    )
    text = ing.process_file("f.txt", use_unstructured=False, return_elements=True)
    assert DummyParser.called == (False, True)
    assert text == "a\nb"


def test_process_file_pdf_fallback(monkeypatch):
    class DummyPDF(ing.PDFParser):
        def parse(self, file_path, high_res=False, ocr=False, return_pages=False):
            return ""

    dummy_pdf2 = types.ModuleType("pdf2image")
    dummy_pdf2.convert_from_path = lambda fp: ["i1", "i2"]
    dummy_pyt = types.ModuleType("pytesseract")
    dummy_pyt.image_to_string = lambda img, lang=None: f"t{img}"
    monkeypatch.setitem(sys.modules, "pdf2image", dummy_pdf2)
    monkeypatch.setitem(sys.modules, "pytesseract", dummy_pyt)
    monkeypatch.setattr(ing, "determine_parser", lambda *a, **k: DummyPDF())
    monkeypatch.setattr(ing, "load_config", lambda: {"ingest": {"ocr_lang": "en"}})
    text = ing.process_file("doc.pdf", ocr=True)
    assert text == "ti1\nti2"


def test_process_file_return_modes(monkeypatch):
    class Dummy:
        def parse(self, f, **kw):
            return ("ok", ["p1"])

    monkeypatch.setattr(ing, "determine_parser", lambda *a, **k: Dummy())
    monkeypatch.setattr(ing, "load_config", lambda: {})
    text, pages = ing.process_file("x", return_pages=True)
    assert text == "ok" and pages == ["p1"]

    class Dummy2:
        def parse(self, f, use_unstructured=None, return_elements=False):
            return [types.SimpleNamespace(text="a"), types.SimpleNamespace(text="b")]

    monkeypatch.setattr(ing, "determine_parser", lambda *a, **k: Dummy2())
    out = ing.process_file("y", return_elements=True)
    assert out == "a\nb"


def test_ingest_into_dataset_options(monkeypatch, tmp_path):
    dataset = DummyDataset()

    class DummyParser:
        def parse(self, file_path, **kw):
            return "txt"

    import datacreek.parsers as parsers

    monkeypatch.setattr(parsers, "WhisperAudioParser", DummyParser)
    monkeypatch.setattr(parsers, "YouTubeParser", DummyParser)
    monkeypatch.setattr(ing, "determine_parser", lambda *a, **k: DummyParser())
    calls = {}

    def fake_process(file_path, config=None, *, high_res=False, **kw):
        calls["high_res"] = high_res
        return "text"

    monkeypatch.setattr(ing, "process_file", fake_process)
    monkeypatch.setattr(ing, "to_kg", lambda *a, **k: None)

    f = tmp_path / "sound.wav"
    f.write_text("x")
    options = ing.IngestOptions(high_res=True)
    ing.ingest_into_dataset(str(f), dataset, high_res=False, options=options)
    assert calls["high_res"] is True
    assert dataset.add_audio_calls
