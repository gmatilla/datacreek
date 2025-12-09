import os

from datacreek.analysis.rollback import SheafSLA, rollback_gremlin_diff


def test_rollback_diff(tmp_path):
    path = rollback_gremlin_diff(repo=os.getcwd(), output="test.diff")
    assert os.path.exists(path)


def test_sheaf_sla_mttr():
    sla = SheafSLA(threshold_hours=1)
    sla.record_failure(0)
    sla.record_failure(3600)
    assert sla.mttr_hours() == 1
    assert sla.sla_met()
