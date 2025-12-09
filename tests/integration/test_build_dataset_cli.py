import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from datacreek.core.scripts import build_dataset


def test_build_dataset_dry_run(monkeypatch):
    called = {}

    def fake_run(source, config, out, dry_run=False):
        called["args"] = (source, config, out, dry_run)

    monkeypatch.setattr(build_dataset, "run_pipeline", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_dataset.py",
            "--source",
            "input.pdf",
            "--config",
            "cfg.yaml",
            "--output",
            "out.txt",
            "--dry-run",
        ],
    )
    build_dataset.main()
    assert called["args"] == ("input.pdf", "cfg.yaml", "out.txt", True)
