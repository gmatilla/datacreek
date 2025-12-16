#!/usr/bin/env python3
"""Git hook preventing breaking changes to the LakeFS schema.

The script compares the current ``.lakefs/schema.yaml`` against the version
committed in ``HEAD``. Removing existing fields or changing their type is
considered a breaking change and will abort the commit.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

SCHEMA_PATH = Path(".lakefs/schema.yaml")


def load_schema(path: Path) -> Dict[str, Any]:
    """Load a YAML schema from ``path`` returning an empty dict if missing."""
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def load_previous_schema(path: Path) -> Optional[Dict[str, Any]]:
    """Return the schema tracked in ``HEAD`` or ``None`` if it does not exist."""
    try:
        res = subprocess.run(
            ["git", "show", f"HEAD:{path.as_posix()}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    return yaml.safe_load(res.stdout) or {}


def is_breaking_change(curr: Dict[str, Any], prev: Optional[Dict[str, Any]]) -> bool:
    """Return ``True`` if removing fields or changing types in ``curr``."""
    if not prev:
        return False
    prev_fields = {f["name"]: f["type"] for f in prev.get("fields", [])}
    curr_fields = {f["name"]: f["type"] for f in curr.get("fields", [])}
    for name, typ in prev_fields.items():
        if name not in curr_fields:
            return True
        if curr_fields[name] != typ:
            return True
    return False


def main() -> int:
    """Entry point for the pre-commit hook."""
    curr = load_schema(SCHEMA_PATH)
    prev = load_previous_schema(SCHEMA_PATH)
    if is_breaking_change(curr, prev):
        print(
            "Breaking schema change detected: remove or modification of existing fields.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
