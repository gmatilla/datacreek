import sys
import types

from typer.testing import CliRunner

from datacreek import cli


def test_init_db(monkeypatch):
    called = {}
    fake_module = types.SimpleNamespace(init_db=lambda: called.setdefault("init", True))
    monkeypatch.setitem(sys.modules, "datacreek.db", fake_module)
    runner = CliRunner()
    result = runner.invoke(cli.app_cli, ["init-db-cmd"])
    assert result.exit_code == 0
    assert called.get("init")
    assert "Database initialized" in result.output


def test_test_cmd(monkeypatch):
    def fake_main(args):
        assert args == ["-q"]
        return 0

    monkeypatch.setitem(sys.modules, "pytest", types.SimpleNamespace(main=fake_main))
    runner = CliRunner()
    result = runner.invoke(cli.app_cli, ["test"])
    assert result.exit_code == 0
