"""Smoke tests running the training CLI on common datasets."""

import json
from pathlib import Path

import cli.train as cli_train


def _run_cli(monkeypatch, tmp_path: Path, samples, expected_trainer: str) -> None:
    data_file = tmp_path / "data.jsonl"
    with data_file.open("w", encoding="utf-8") as fh:
        for rec in samples:
            fh.write(json.dumps(rec) + "\n")

    called = {}
    monkeypatch.setattr(
        cli_train,
        "load_model",
        lambda model_id, bits=4, max_seq=8192: object(),
    )
    monkeypatch.setattr(
        cli_train,
        "add_lora",
        lambda model, r, alpha, target_modules: model,
    )

    def fake_build_trainer(name, model, dataset, **kwargs):
        called["trainer"] = name
        called["epochs"] = kwargs.get("epochs")

        class DummyTrainer:
            def train(self):
                called["trained"] = True

        return DummyTrainer()

    monkeypatch.setattr(cli_train, "build_trainer", fake_build_trainer)

    cli_train.main(
        [
            "--model",
            "tiny",
            "--dataset-path",
            str(data_file),
            "--task",
            expected_trainer,
            "--epochs",
            "1",
        ]
    )

    assert called["trainer"] == expected_trainer
    assert called["trained"]
    assert called["epochs"] == 1


def test_tinystories_generation_epoch(monkeypatch, tmp_path):
    """Run one epoch on a TinyStories-style generation dataset."""
    samples = [
        {"prompt": "Tell a story about a dragon", "response": "Once upon a time"}
    ]
    _run_cli(monkeypatch, tmp_path, samples, "generation")


def test_dbpedia_classification_epoch(monkeypatch, tmp_path):
    """Run one epoch on a DBPedia-style classification dataset."""
    samples = [{"text": "Ford produces cars", "label": 1}]
    _run_cli(monkeypatch, tmp_path, samples, "classification")
