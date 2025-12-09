import os
import time

from datacreek.core.knowledge_graph import (
    get_cleanup_cfg,
    start_cleanup_watcher,
    stop_cleanup_watcher,
)


def test_cleanup_watcher_reload(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "cleanup:\n  sigma: 0.5\n  tau: 2\n  k_min: 3\n  lp_sigma: 0.2\n  lp_topk: 4\n  hub_deg: 10\n"
    )
    os.environ["DATACREEK_CONFIG"] = str(cfg_file)
    stop_cleanup_watcher()
    start_cleanup_watcher()
    time.sleep(0.2)
    vals = get_cleanup_cfg()
    assert vals["sigma"] == 0.5
    cfg_file.write_text(
        "cleanup:\n  sigma: 0.8\n  tau: 1\n  k_min: 5\n  lp_sigma: 0.1\n  lp_topk: 3\n  hub_deg: 20\n"
    )
    time.sleep(0.2)
    vals = get_cleanup_cfg()
    stop_cleanup_watcher()
    assert vals["sigma"] == 0.8


def test_default_config_reload(caplog, tmp_path):
    """Cleanup thresholds should update when the default YAML changes."""
    from pathlib import Path

    import yaml

    cfg_src = Path("configs/default.yaml")
    cfg_file = tmp_path / "default.yaml"
    cfg_file.write_text(cfg_src.read_text())
    os.environ["DATACREEK_CONFIG"] = str(cfg_file)
    stop_cleanup_watcher()
    caplog.set_level("INFO")
    start_cleanup_watcher()
    time.sleep(0.3)
    orig = get_cleanup_cfg()["tau"]
    cfg = yaml.safe_load(cfg_file.read_text())
    cfg["cleanup"]["tau"] = orig + 1
    cfg_file.write_text(yaml.dump(cfg))
    time.sleep(0.3)
    vals = get_cleanup_cfg()
    stop_cleanup_watcher()
    assert vals["tau"] == orig + 1
    assert any("cleanup thresholds updated" in r.message for r in caplog.records)
