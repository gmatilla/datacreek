from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def export_delta(
    result: list | dict | str, *, root: str, org_id: int | str, kind: str
) -> Path:
    """Save ``result`` under a Delta Lake style partition and return the file path."""
    date = datetime.now(timezone.utc).date().isoformat()
    path = Path(root) / f"org_id={org_id}" / f"kind={kind}" / f"date={date}"
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / "data.jsonl"
    with open(file_path, "w", encoding="utf-8") as fh:
        if isinstance(result, str):
            fh.write(result)
        else:
            if isinstance(result, dict):
                lines = [json.dumps(result, ensure_ascii=False)]
            else:
                lines = [json.dumps(r, ensure_ascii=False) for r in result]
            fh.write("\n".join(lines))
    return file_path


def lakefs_commit(path: Path, repo: str) -> str | None:
    """Run ``lakefs commit`` for ``repo`` pointing at ``path`` and return the commit SHA."""
    try:
        res = subprocess.run(
            ["lakefs", "commit", repo, "-m", f"export {path}", "--json"],
            check=True,
            capture_output=True,
            text=True,
        )
        try:
            data = json.loads(res.stdout)
            return data.get("id")
        except Exception:
            logger.debug("failed to parse commit id: %s", res.stdout)
            return None
    except Exception:  # pragma: no cover - treat commit errors as non fatal
        logger.exception("LakeFS commit failed")
        return None


def delta_optimize(root: Path) -> None:
    """Run ``delta optimize`` using ZORDER by ``org_id`` and ``kind`` columns."""
    try:
        subprocess.run(
            ["delta", "optimize", str(root), "--zorder-by", "org_id,kind"],
            check=True,
        )
    except Exception:  # pragma: no cover - not fatal
        logger.exception("Delta optimize failed")


def delta_vacuum(root: Path, retain_days: int = 30) -> None:
    """Run ``delta vacuum`` retaining files for ``retain_days`` days."""
    try:
        subprocess.run(
            ["delta", "vacuum", str(root), "--retain", str(retain_days)],
            check=True,
        )
    except Exception:  # pragma: no cover - not fatal
        logger.exception("Delta vacuum failed")
