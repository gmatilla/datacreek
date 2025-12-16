from contextlib import contextmanager
from typing import Generator, Tuple

from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def create_progress(description: str, total: int) -> Tuple[Progress, int]:
    """Return a ``Progress`` instance and task ID for ``description``."""
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    task_id = progress.add_task(description, total=total)
    return progress, task_id


@contextmanager
def progress_context(
    description: str, total: int
) -> Generator[Tuple[Progress, int], None, None]:
    """Yield a started progress bar and stop it afterwards."""
    progress, task_id = create_progress(description, total)
    progress.start()
    try:
        yield progress, task_id
    finally:
        progress.stop()
