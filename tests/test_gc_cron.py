"""Validate configuration of the GC checkpoint CronJob."""

from pathlib import Path

import yaml


def test_gc_cron_schedule_and_command():
    """CronJob should run daily and invoke the cleanup script."""
    manifest = yaml.safe_load(Path("k8s/cron/gc.yaml").read_text())
    # Ensure the job triggers once per day
    assert manifest["spec"]["schedule"] == "@daily"
    container = manifest["spec"]["jobTemplate"]["spec"]["template"]["spec"][
        "containers"
    ][0]
    cmd = container.get("args") or container.get("command") or []
    assert any("cleanup_checkpoints.py" in part for part in cmd)
