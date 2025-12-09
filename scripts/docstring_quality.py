#!/usr/bin/env python3
"""Check docstring coverage using ``interrogate``.

This script replicates the ``docstring-quality`` tool which normally ships
with the ``docstring-checker`` project.  It parses the coverage reported by
``interrogate`` and exits with ``1`` when the measured value is below the
configured threshold.
"""
import argparse
import re
import subprocess
import sys

DEFAULT_TARGET = 0.80


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fail-under",
        type=float,
        default=DEFAULT_TARGET,
        help="fail if docstring coverage is below this fraction",
    )
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "interrogate",
        "-e",
        "datacreek/api.py",
        "-e",
        "datacreek/db.py",
        "-e",
        "datacreek/pipelines.py",
        "-e",
        "datacreek/schemas.py",
        "-e",
        "datacreek/services.py",
        "-e",
        "datacreek/storage.py",
        "-e",
        "datacreek/tasks.py",
        "-e",
        "datacreek/config_models.py",
        "-e",
        "datacreek/utils",
        "datacreek",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout.strip().splitlines()
    coverage = None
    for line in output:
        match = re.search(r"actual: ([0-9.]+)%", line)
        if match:
            coverage = float(match.group(1)) / 100.0
            break
    if coverage is None:
        print("Failed to parse coverage")
        print(proc.stdout)
        print(proc.stderr)
        return 1
    print(f"docstring quality: {coverage:.2f}")
    return 0 if coverage >= args.fail_under else 1


if __name__ == "__main__":
    raise SystemExit(main())
