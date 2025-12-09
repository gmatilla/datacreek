import builtins
import sys
import types

import pytest

from datacreek.parsers.pdf_parser import PDFParser


class DummyEl:
    def __init__(self, text=None):
        self.text = text


def test_pdf_parser_unstructured(monkeypatch):
    def fake_partition_pdf(filename):
        return [DummyEl("a"), DummyEl("b")]

    monkeypatch.setitem(
        sys.modules,
        "unstructured.partition.pdf",
        types.SimpleNamespace(partition_pdf=fake_partition_pdf),
    )
    parser = PDFParser()
    assert parser.parse("f.pdf") == "a\nb"


def test_pdf_parser_high_res(monkeypatch):
    class FakeParse:
        def parse(self, path):
            return "hi"

    monkeypatch.setitem(
        sys.modules, "llamaparse", types.SimpleNamespace(LlamaParse=lambda: FakeParse())
    )
    parser = PDFParser()
    assert parser.parse("f.pdf", high_res=True) == "hi"


def test_pdf_parser_ocr(monkeypatch):
    # partition_pdf returns empty text to trigger OCR
    def fake_partition_pdf(filename):
        return [DummyEl(None)]

    monkeypatch.setitem(
        sys.modules,
        "unstructured.partition.pdf",
        types.SimpleNamespace(partition_pdf=fake_partition_pdf),
    )

    def fake_convert(path):
        return ["img1", "img2"]

    def fake_image_to_string(img):
        return f"text-{img}"

    monkeypatch.setitem(
        sys.modules, "pdf2image", types.SimpleNamespace(convert_from_path=fake_convert)
    )
    monkeypatch.setitem(
        sys.modules,
        "pytesseract",
        types.SimpleNamespace(image_to_string=fake_image_to_string),
    )
    parser = PDFParser()
    text, pages = parser.parse("f.pdf", ocr=True, return_pages=True)
    assert "text-img1" in text and "text-img2" in text
    assert pages == text.split("\f")


def test_pdf_parser_import_error(monkeypatch):
    def fake_import(name, *args, **kwargs):
        if name == "llamaparse":
            raise ImportError("x")
        return orig_import(name, *args, **kwargs)

    orig_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", fake_import)
    parser = PDFParser()
    with pytest.raises(ImportError):
        parser.parse("f.pdf", high_res=True)
    monkeypatch.setattr(builtins, "__import__", orig_import)
