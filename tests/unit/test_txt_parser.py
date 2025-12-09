import os
import tempfile

from datacreek.parsers.txt_parser import TXTParser


def test_txt_parser_reads_file(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text("hello")
    parser = TXTParser()
    assert parser.parse(str(p)) == "hello"
