"""Pre-commit hook verifying Jinja templates have matching SHA256 headers.

The hook expects each template to begin with a line of the form::

    # sha256: <hex digest>

The digest must correspond to the SHA256 of the remainder of the file
following the first newline.  This prevents unnoticed edits to a template
without bumping its recorded hash used for dataset reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Iterable

HEADER_RE = re.compile(r"^# sha256: ([0-9a-f]{64})$")


def _compute_digest(path: Path) -> str:
    """Compute the SHA256 of a template body excluding the first line.

    Parameters
    ----------
    path:
        Path to the template file.

    Returns
    -------
    str
        Hexadecimal digest of the template body.
    """
    with path.open("rb") as fh:
        # Read and discard the header line (first line)
        first_line = fh.readline()
        body = fh.read()

    match = HEADER_RE.match(first_line.decode("utf-8", "replace").strip())
    if match is None:
        raise ValueError(f"{path} is missing '# sha256: <digest>' header")

    digest = hashlib.sha256(body).hexdigest()
    expected = match.group(1)
    if digest != expected:
        raise ValueError(
            f"{path} digest mismatch: header {expected} but actual {digest}"
        )
    return digest


def validate_paths(paths: Iterable[Path]) -> None:
    """Validate SHA256 headers for provided template paths."""
    errors: list[str] = []
    for path in paths:
        try:
            _compute_digest(path)
        except ValueError as exc:  # collect errors but continue checking
            errors.append(str(exc))
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        raise SystemExit(1)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Template files to verify")
    args = parser.parse_args(list(argv) if argv is not None else None)
    validate_paths(args.paths)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
