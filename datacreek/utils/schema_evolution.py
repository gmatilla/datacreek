"""Database schema evolution helpers."""

from __future__ import annotations

from typing import Iterable

from sqlalchemy import Engine, inspect, text

__all__ = ["add_column_if_missing"]


def add_column_if_missing(
    engine: Engine,
    table: str,
    column: str,
    col_type: str,
) -> None:
    """Add ``column`` to ``table`` if it does not already exist.

    Parameters
    ----------
    engine:
        SQLAlchemy :class:`~sqlalchemy.engine.Engine` connected to the DB.
    table:
        Name of the table to alter.
    column:
        Column name to add.
    col_type:
        SQL type expression, e.g. ``"TEXT"`` or ``"INTEGER"``.

    The function inspects the table's existing columns and issues an
    ``ALTER TABLE`` statement only when the column is missing. This is a
    lightweight helper for scripted schema evolution tasks.
    """

    insp = inspect(engine)
    names: Iterable[str] = (c["name"] for c in insp.get_columns(table))
    if column in names:
        return
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))
