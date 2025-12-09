from pathlib import Path

from datacreek.core.context import AppContext
from datacreek.utils.config import DEFAULT_CONFIG_PATH


def test_appcontext_defaults():
    ctx = AppContext()
    assert ctx.config_path == Path(DEFAULT_CONFIG_PATH)
    assert ctx.config == {}


def test_appcontext_custom_path(tmp_path):
    custom = tmp_path / "cfg.yaml"
    ctx = AppContext(custom)
    assert ctx.config_path == custom


def test_appcontext_env_override(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}")
    monkeypatch.setenv("DATACREEK_CONFIG", str(cfg))

    def fake_load(path):
        assert path == str(cfg)
        return {"ok": True}

    monkeypatch.setattr("datacreek.core.context.load_config", fake_load)
    ctx = AppContext()
    assert ctx.config_path == cfg
    res = ctx.load()
    assert res == {"ok": True}


def test_appcontext_param_wins(tmp_path, monkeypatch):
    env_cfg = tmp_path / "env.json"
    env_cfg.write_text("{}")
    monkeypatch.setenv("DATACREEK_CONFIG", str(env_cfg))
    param_cfg = tmp_path / "param.json"

    def fake_load(path):
        return {"path": path}

    monkeypatch.setattr("datacreek.core.context.load_config", fake_load)
    ctx = AppContext(param_cfg)
    assert ctx.config_path == param_cfg
    assert ctx.load() == {"path": str(param_cfg)}
