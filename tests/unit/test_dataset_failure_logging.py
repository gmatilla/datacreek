import logging

from datacreek.core import dataset_full


def test_log_ignored_failure_records_warning(caplog):
    caplog.set_level(logging.WARNING)

    dataset_full._log_ignored_failure("ignored failure path")

    assert "ignored failure path" in caplog.text
    assert "continuing despite failure" in caplog.text
