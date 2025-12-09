import json
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))
import cli.train as cli_train


def test_help_shows_options():
    result = subprocess.run(
        [sys.executable, "cli/train.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--model" in result.stdout
    assert "--dataset-path" in result.stdout
    assert "--epochs" in result.stdout


def test_main_runs_with_config(monkeypatch, tmp_path):
    data_file = tmp_path / "data.jsonl"
    data_file.write_text(json.dumps({"text": "hello"}) + "\n", encoding="utf-8")

    cfg_file = tmp_path / "config.yml"
    yaml.safe_dump({"task": "generation", "bits": 8, "epochs": 2}, cfg_file.open("w"))

    called = {}

    def fake_load_model(model_id, bits=4, max_seq=8192):
        called["bits"] = bits
        return object()

    monkeypatch.setattr(cli_train, "load_model", fake_load_model)
    monkeypatch.setattr(
        cli_train, "add_lora", lambda model, r, alpha, target_modules: model
    )
    monkeypatch.setattr(cli_train, "detect_task", lambda ds: "generation")
    monkeypatch.setattr(cli_train, "_format_dataset", lambda task, ds: ds)

    class DummyTrainer:
        def __init__(self):
            self.trained = False

        def train(self):
            called["trained"] = True

    def fake_build_trainer(task, model, dataset, **kwargs):
        called["epochs"] = kwargs.get("epochs")
        return DummyTrainer()

    monkeypatch.setattr(cli_train, "build_trainer", fake_build_trainer)

    cli_train.main(
        [
            "--model",
            "tiny",
            "--dataset-path",
            str(data_file),
            "--config",
            str(cfg_file),
        ]
    )

    assert called["bits"] == 8
    assert called["trained"]
    assert called["epochs"] == 2
