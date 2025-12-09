# pragma: no cover
# CLI wrapper for building datasets, excluded from coverage
# because it orchestrates many external dependencies.
"""Convenience entry point for the dataset pipeline."""

from datacreek.core.scripts import build_dataset

if __name__ == "__main__":
    build_dataset.main()
