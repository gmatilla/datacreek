import os
import subprocess
import time
from pathlib import Path


def _touch(path: Path, mtime: float) -> None:
    """Create ``path`` and set its modification time."""
    path.write_text("x")
    os.utime(path, (mtime, mtime))


def test_gc_removes_old_non_best(tmp_path: Path) -> None:
    """Old checkpoints without the ``best`` prefix should be deleted."""
    now = time.time()
    old = tmp_path / "old.ckpt"
    best_old = tmp_path / "best-old.ckpt"
    recent = tmp_path / "recent.ckpt"

    # Simulate files with different ages
    thirty_one_days = 31 * 24 * 3600
    _touch(old, now - thirty_one_days)
    _touch(best_old, now - thirty_one_days)
    _touch(recent, now - 5 * 24 * 3600)

    subprocess.run(
        [
            "bash",
            str(Path(__file__).resolve().parents[2] / "scripts" / "gc_checkpoints.sh"),
            str(tmp_path),
        ],
        check=True,
    )

    assert not old.exists(), "old checkpoint should be removed"
    assert best_old.exists(), "best checkpoint must be kept"
    assert recent.exists(), "recent checkpoint must be kept"
