"""Simple CLI to launch fine-tuning with Unsloth and TRL.

The script wires together the building blocks defined in :mod:`training`.
It loads a base model with optional 4-bit quantization and LoRA adapters,
formats a JSONL dataset, selects an appropriate trainer, and kicks off
training.  A YAML configuration file can supply defaults for any CLI option.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Ensure repository root is on sys.path when running as a script
sys.path.append(str(Path(__file__).resolve().parents[1]))

from training import (
    add_lora,
    build_trainer,
    detect_task,
    format_classif,
    format_rlhf,
    format_sft,
    load_model,
)

FORMATTERS = {
    "generation": format_sft,
    "qa": format_sft,
    "classification": format_classif,
    "rlhf_ppo": format_rlhf,
    "rlhf_dpo": format_rlhf,
}


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Fine-tune language models using Unsloth + TRL pipeline",
    )
    parser.add_argument("--model", required=True, help="Model identifier or path")
    parser.add_argument("--task", help="Task type if known (auto-detected otherwise)")
    parser.add_argument("--dataset-path", required=True, help="Path to JSONL dataset")
    parser.add_argument(
        "--trainer",
        help="Override trainer selection (defaults to detected task)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=16.0,
        help="LoRA alpha scaling factor forwarded to the trainer",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Generic hyperparameter forwarded to the trainer",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="Quantization bits when loading the base model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Optional YAML config file providing defaults",
    )
    return parser


def _apply_config(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> argparse.Namespace:
    """Override default arguments with values from a YAML config file."""
    if args.config:
        with open(args.config, "r") as fh:
            config = yaml.safe_load(fh) or {}
        for key, value in config.items():
            if getattr(args, key, parser.get_default(key)) == parser.get_default(key):
                setattr(args, key, value)
    return args


def _read_dataset(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL dataset from *path*."""
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _format_dataset(task: str, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format *dataset* according to *task* using helpers from :mod:`training`."""
    fmt = FORMATTERS.get(task)
    if fmt:
        return [fmt(sample) for sample in dataset]
    return dataset


def main(cli_args: List[str] | None = None) -> None:
    """Entry point for the training CLI."""
    parser = _build_parser()
    args = parser.parse_args(cli_args)
    args = _apply_config(args, parser)

    dataset = _read_dataset(args.dataset_path)
    task = args.task or detect_task(dataset)
    dataset = _format_dataset(task, dataset)

    model = load_model(args.model, bits=args.bits)
    model = add_lora(model, r=8, alpha=args.alpha, target_modules=["q_proj", "v_proj"])

    trainer_name = args.trainer or task
    trainer = build_trainer(
        trainer_name,
        model,
        dataset,
        alpha=args.alpha,
        beta=args.beta,
        epochs=args.epochs,
    )
    trainer.train()


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
