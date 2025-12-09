import json

import datacreek.templates.library as lib


def test_load_templates(tmp_path, monkeypatch):
    spec = tmp_path / "foo.json"
    spec.write_text(
        json.dumps({"schema": {"type": "object"}, "max_length": 5, "regex": r".*"})
    )
    monkeypatch.setattr(lib, "TEMPLATE_DIR", tmp_path)
    templates = lib.load_templates()
    assert "foo" in templates
    assert templates["foo"].max_length == 5


def test_validate_output(monkeypatch):
    tmpl = lib.PromptTemplate("t", {"type": "object"}, 10, r".*")
    monkeypatch.setitem(lib.TEMPLATES, "t", tmpl)
    assert lib.validate_output("t", "{}")
    monkeypatch.setitem(lib.TEMPLATES, "t", tmpl)
    assert not lib.validate_output("t", '{"a":')
