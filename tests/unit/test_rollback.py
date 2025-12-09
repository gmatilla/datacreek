import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from datacreek.analysis import rollback


def _init_repo(tmpdir):
    subprocess.check_call(["git", "init"], cwd=tmpdir)
    # config user
    subprocess.check_call(["git", "config", "user.email", "a@b.c"], cwd=tmpdir)
    subprocess.check_call(["git", "config", "user.name", "test"], cwd=tmpdir)
    # first commit
    (Path(tmpdir) / "file.txt").write_text("hello")
    subprocess.check_call(["git", "add", "file.txt"], cwd=tmpdir)
    subprocess.check_call(["git", "commit", "-m", "first"], cwd=tmpdir)
    # second commit
    (Path(tmpdir) / "file.txt").write_text("world")
    subprocess.check_call(["git", "add", "file.txt"], cwd=tmpdir)
    subprocess.check_call(["git", "commit", "-m", "second"], cwd=tmpdir)


def test_rollback_gremlin_diff(tmp_path):
    _init_repo(tmp_path)
    diff_path = rollback.rollback_gremlin_diff(str(tmp_path))
    assert os.path.isfile(diff_path)
    content = open(diff_path).read()
    assert "-hello" in content and "+world" in content


def test_sheaf_sla():
    sla = rollback.SheafSLA(threshold_hours=1.0)
    now = 1000.0
    sla.record_failure(now)
    sla.record_failure(now + 3600)
    assert pytest.approx(sla.mttr_hours(), abs=1e-6) == 1.0
    assert sla.sla_met()
    sla.record_failure(now + 3 * 3600)
    assert not sla.sla_met()
