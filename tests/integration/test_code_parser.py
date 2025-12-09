from datacreek.parsers.code_parser import CodeParser


def test_code_parser(tmp_path):
    f = tmp_path / "sample.py"
    f.write_text("print('hi')")
    parser = CodeParser()
    assert parser.parse(str(f)) == "print('hi')"
