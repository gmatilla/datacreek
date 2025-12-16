import typer

app_cli = typer.Typer(
    help="Maintenance utilities for datacreek (database init, tests, upgrades)"
)


@app_cli.command()
def init_db_cmd() -> None:
    """Create database tables."""
    from datacreek.db import init_db

    init_db()
    typer.echo("Database initialized")


@app_cli.command()
def test():
    """Run the unit test suite."""
    import pytest

    raise SystemExit(pytest.main(["-q"]))


if __name__ == "__main__":
    app_cli()
