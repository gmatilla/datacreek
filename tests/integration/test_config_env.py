import importlib
from pathlib import Path

import datacreek.utils.config as cfg


def test_env_config_path(monkeypatch, tmp_path):
    custom = tmp_path / "config.yaml"
    custom.write_text("database:\n  url: sqlite:///tmp.db\n")
    monkeypatch.setenv("DATACREEK_CONFIG", str(custom))
    importlib.reload(cfg)
    config = cfg.load_config()
    assert config["database"]["url"] == "sqlite:///tmp.db"
