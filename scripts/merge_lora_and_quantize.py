#!/usr/bin/env python3
"""Merge LoRA weights into a base model and quantize to NF4.

This small CLI utility loads two ``state_dict`` files, applies the LoRA
updates to the base weights and exports an NF4 quantized ``model.gguf`` file.
The resulting file begins with the magic header ``b"GGUF"`` followed by a
``torch.save`` of the quantized parameters produced by
:func:`training.quant_utils.quantize_state_dict_nf4`.
"""

from __future__ import annotations

import argparse

from training.quant_utils import merge_and_quantize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="path to base model state_dict")
    parser.add_argument("--lora", required=True, help="path to LoRA state_dict")
    parser.add_argument(
        "--out",
        default="model.gguf",
        help="output GGUF file containing NF4 quantized weights",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_and_quantize(args.base, args.lora, args.out)


if __name__ == "__main__":
    main()
