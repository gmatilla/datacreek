import random
import subprocess
import sys
import types
from pathlib import Path

import pytest

from datacreek.analysis import explain_viz, ingestion, privacy, rollback


def test_partition_files_to_atoms_fallback(tmp_path, monkeypatch):
    path = tmp_path / "f.txt"
    path.write_text("a\nb\n\nc\n")
    # stub unstructured.partition.auto.partition to raise
    mod = types.SimpleNamespace(
        partition=lambda p: (_ for _ in ()).throw(RuntimeError())
    )
    monkeypatch.setitem(sys.modules, "unstructured.partition.auto", mod)
    atoms = ingestion.partition_files_to_atoms(str(path))
    assert atoms == ["a", "b", "c"]


def test_partition_files_to_atoms_unstructured(monkeypatch, tmp_path):
    class E:
        def __init__(self, t):
            self.text = t

    monkeypatch.setitem(
        sys.modules,
        "unstructured.partition.auto",
        types.SimpleNamespace(partition=lambda p: [E("x"), E(" y "), E("")]),
    )
    res = ingestion.partition_files_to_atoms("irrelevant")
    assert res == ["x", "y"]


def test_parse_code_to_atoms(tmp_path):
    code = """\
def f(x):
    return x + 1

class C:
    pass
"""
    p = tmp_path / "sample.py"
    p.write_text(code)
    atoms = ingestion.parse_code_to_atoms(str(p))
    joined = "\n".join(atoms)
    assert "def f" in joined and "class C" in joined


def test_parse_code_to_atoms_fallback(tmp_path, monkeypatch):
    p = tmp_path / "a.py"
    p.write_text("print('hi')\n")
    monkeypatch.setitem(
        sys.modules,
        "ast",
        types.SimpleNamespace(parse=lambda x: (_ for _ in ()).throw(RuntimeError())),
    )
    atoms = ingestion.parse_code_to_atoms(str(p))
    assert atoms == ["print('hi')"]


def test_transcribe_audio(monkeypatch):
    stub_model = types.SimpleNamespace(transcribe=lambda path: {"text": "hello"})
    whisper = types.SimpleNamespace(load_model=lambda name: stub_model)
    monkeypatch.setitem(sys.modules, "whisper", whisper)
    assert ingestion.transcribe_audio("x.wav") == "hello"


def test_blip_caption_image(monkeypatch):
    class FakeImage:
        pass

    class FakeModel:
        def generate(self, **kw):
            return [0]

    class FakeProcessor:
        def __init__(self):
            pass

        def __call__(self, img, return_tensors="pt"):
            return {}

        def decode(self, ids, skip_special_tokens=True):
            return "caption"

        @classmethod
        def from_pretrained(cls, name):
            return cls()

    monkeypatch.setitem(
        sys.modules,
        "PIL",
        types.SimpleNamespace(Image=types.SimpleNamespace(open=lambda p: FakeImage())),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            BlipForConditionalGeneration=types.SimpleNamespace(
                from_pretrained=lambda n: FakeModel()
            ),
            BlipProcessor=FakeProcessor,
        ),
    )
    assert ingestion.blip_caption_image("img.png") == "caption"


def test_rollback_gremlin_diff(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
    f = repo / "f.txt"
    f.write_text("a")
    subprocess.run(
        ["git", "add", "f.txt"], cwd=repo, check=True, stdout=subprocess.DEVNULL
    )
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL
    )
    f.write_text("b")
    subprocess.run(
        ["git", "commit", "-am", "change"],
        cwd=repo,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    diff = rollback.rollback_gremlin_diff(str(repo))
    assert Path(diff).exists() and Path(diff).read_text()


def test_sheaf_sla():
    sla = rollback.SheafSLA(threshold_hours=1)
    sla.record_failure(1000.0)
    sla.record_failure(4600.0)
    assert pytest.approx(sla.mttr_hours(), 0.01) == 1.0
    assert sla.sla_met()


def test_explain_to_svg_basic():
    data = {"nodes": ["A", "B"], "edges": [("A", "B")], "attention": {"A->B": 0.8}}
    svg = explain_viz.explain_to_svg(data)
    assert svg.startswith("<svg") and "A" in svg and "B" in svg


def test_k_out_randomized_response(monkeypatch):
    seq = iter([0.0, 0.9])
    monkeypatch.setattr(random, "random", lambda: next(seq))
    items = ["x", "y"]
    res = privacy.k_out_randomized_response(items, k=1)
    # first kept, second replaced
    assert res[0] == "x" and res[1] in items
