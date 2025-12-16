#!/usr/bin/env python
"""Rebuild embeddings and FAISS index from a dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datacreek.core.dataset import DatasetBuilder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="Path to dataset JSON file")
    parser.add_argument(
        "--from-version",
        dest="version",
        default="1.0",
        help="Embedding version to record",
    )
    args = parser.parse_args()
    data = json.loads(Path(args.dataset).read_text())
    ds = DatasetBuilder.from_dict(data)
    ds.compute_graph_embeddings()
    ds.build_faiss_index(embedding_version=args.version)
    Path(args.dataset).write_text(json.dumps(ds.to_dict()))


if __name__ == "__main__":
    main()
