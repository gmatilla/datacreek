import sys
import types

from datacreek.parsers.docx_parser import DOCXParser
from datacreek.parsers.html_parser import HTMLParser
from datacreek.parsers.pdf_parser import PDFParser


def test_docx_parser_unstructured(monkeypatch, tmp_path):
    f = tmp_path / "sample.docx"
    f.write_bytes(b"doc")
    module = types.ModuleType("unstructured.partition.docx")
    module.partition_docx = lambda filename: [types.SimpleNamespace(text="hello")]
    monkeypatch.setitem(sys.modules, "unstructured.partition.docx", module)
    parser = DOCXParser()
    assert parser.parse(str(f), use_unstructured=True) == "hello"


def test_pdf_parser_unstructured(monkeypatch, tmp_path):
    f = tmp_path / "sample.pdf"
    f.write_bytes(b"pdf")
    module = types.ModuleType("unstructured.partition.pdf")
    module.partition_pdf = lambda filename: [types.SimpleNamespace(text="hi")]
    monkeypatch.setitem(sys.modules, "unstructured.partition.pdf", module)
    parser = PDFParser()
    assert parser.parse(str(f), use_unstructured=True) == "hi"


def test_html_parser_unstructured(monkeypatch, tmp_path):
    f = tmp_path / "page.html"
    f.write_text("<html><body><p>Hello</p></body></html>")
    module = types.ModuleType("unstructured.partition.html")
    module.partition_html = lambda url=None, filename=None: [
        types.SimpleNamespace(text="hi html")
    ]
    monkeypatch.setitem(sys.modules, "unstructured.partition.html", module)
    parser = HTMLParser()
    assert parser.parse(str(f), use_unstructured=True) == "hi html"
