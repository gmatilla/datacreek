import importlib.util
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import types

stub = types.ModuleType("datacreek.config_models")
for name in [
    "CurateSettings",
    "FormatSettings",
    "GenerationSettings",
    "LLMSettings",
    "OpenAISettings",
    "VLLMSettings",
]:
    cls = type(
        name,
        (),
        {"__dataclass_fields__": {}, "from_dict": classmethod(lambda cls, d: cls())},
    )
    setattr(stub, name, cls)
sys.modules["datacreek.config_models"] = stub
spec = importlib.util.spec_from_file_location(
    "dconfig", PROJECT_ROOT / "datacreek" / "utils" / "config.py"
)
config = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(config)
Config = config.Config
start_config_watcher = config.start_config_watcher
stop_config_watcher = config.stop_config_watcher


def test_config_hot_reload(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("fractal:\n  bootstrap_seed: 1\n")
    os.environ["DATACREEK_CONFIG"] = str(cfg_file)
    stop_config_watcher()
    start_config_watcher()
    config.Config.reload()
    time.sleep(0.2)
    assert Config.get()["fractal"]["bootstrap_seed"] == 1
    cfg_file.write_text("fractal:\n  bootstrap_seed: 7\n")
    time.sleep(0.2)
    assert Config.get()["fractal"]["bootstrap_seed"] == 7
    stop_config_watcher()
