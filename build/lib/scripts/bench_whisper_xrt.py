#!/usr/bin/env python3
"""Measure Whisper realtime factor on CPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from datacreek.parsers.whisper_audio_parser import WhisperAudioParser


def run_bench(path: str, loops: int = 1, max_seconds: int = 30) -> dict[str, float]:
    """Return realtime factor for ``loops`` parses of ``path``."""
    parser = WhisperAudioParser()
    start = time.perf_counter()
    for _ in range(loops):
        parser.parse(path)
    duration = time.perf_counter() - start
    xrt = duration / (loops * max_seconds)
    return {"whisper_xrt": float(xrt)}


def main(argv: list[str] | None = None) -> dict[str, float]:
    """CLI entrypoint writing the realtime factor to a file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="samples/mini/sample.wav")
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--max-seconds", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("metrics_prometheus"))
    args = parser.parse_args(argv)
    res = run_bench(args.file, args.loops, args.max_seconds)
    text = f"whisper_xrt {res['whisper_xrt']:.6f}"
    args.output.write_text(text)
    print(text)
    return res


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
