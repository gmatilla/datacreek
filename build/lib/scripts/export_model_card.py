#!/usr/bin/env python3
"""Generate model card JSON and optionally HTML."""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[0] = str(ROOT)
import os

os.environ.setdefault("DATACREEK_REQUIRE_PERSISTENCE", "0")

import argparse
import json

try:
    from jinja2 import Template
except Exception:  # pragma: no cover - optional
    Template = None  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model card metrics")
    parser.add_argument("out", help="Output JSON file")
    parser.add_argument("--html", help="Optional HTML output file")
    args = parser.parse_args()

    card = {
        "bias_wasserstein": 0.0,
        "sigma_db": 0.0,
        "H_wave": 0.0,
        "prune_ratio": 0.0,
        "cca_sha": "none",
    }
    try:
        import subprocess

        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT))
            .decode()
            .strip()
        )
    except Exception:  # pragma: no cover - git missing
        sha = "unknown"
    card["code_commit"] = sha
    with open(args.out, "w") as fh:
        json.dump(card, fh)
    if args.html:
        if Template is None:
            with open(args.html, "w") as fh:
                fh.write(json.dumps(card, indent=2))
        else:
            html = Template(
                """
                <html><body><h1>Model Card</h1>
                <ul>{% for k,v in card.items() %}<li>{{k}}: {{v}}</li>{% endfor %}</ul>
                </body></html>
                """
            ).render(card=card)
            with open(args.html, "w") as fh:
                fh.write(html)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
