"""Tests for the embedding drift Airflow DAG."""

import sys
from pathlib import Path
from types import SimpleNamespace

# Ensure repository root on path for ``dags`` package import during testing
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dags.embedding_drift import compute_drift_task, dag, trigger_retrain_task


def test_dag_structure() -> None:
    """The DAG should consist of two sequential tasks."""

    task_ids = {t.task_id for t in dag.tasks}
    assert task_ids == {"compute_drift", "trigger_retrain"}
    assert dag.get_task("trigger_retrain").upstream_task_ids == {"compute_drift"}


def test_trigger_retrain_branching() -> None:
    """Only critical drift values should set the ``triggered`` flag."""

    class TI(SimpleNamespace):
        def xcom_push(self, key, value):
            self.store = {key: value}

        def xcom_pull(self, key, task_ids):
            return self.data

    # Non-critical drift
    ti = TI()
    compute_drift_task(ti=ti, dag_run=SimpleNamespace(conf={"drift": 0.08}))
    ti.data = ti.store["drift"]
    trigger_retrain_task(ti=ti)
    assert ti.store["triggered"] is False

    # Critical drift
    ti = TI()
    compute_drift_task(ti=ti, dag_run=SimpleNamespace(conf={"drift": 0.12}))
    ti.data = ti.store["drift"]
    trigger_retrain_task(ti=ti)
    assert ti.store["triggered"] is True
