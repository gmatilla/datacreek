import importlib.util
import os
import sys
import time
from pathlib import Path

from datacreek.utils.config import Config, start_config_watcher, stop_config_watcher


def test_config_hot_reload(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("fractal:\n  bootstrap_seed: 1\n")
    os.environ["DATACREEK_CONFIG"] = str(cfg_file)
    stop_config_watcher()
    start_config_watcher()
    Config.reload()
    time.sleep(0.2)
    assert Config.get()["fractal"]["bootstrap_seed"] == 1
    cfg_file.write_text("fractal:\n  bootstrap_seed: 7\n")
    time.sleep(0.2)
    assert Config.get()["fractal"]["bootstrap_seed"] == 7
    stop_config_watcher()
