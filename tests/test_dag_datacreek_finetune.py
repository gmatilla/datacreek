"""Unit tests for the Airflow DAG `datacreek_finetune`.

These tests verify the task graph topology and XCom usage
for propagating the tenant identifier throughout the workflow.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

# Ensure repository root is on path so ``dags`` can be imported when tests run
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dags.datacreek_finetune import (
    build_dataset_task,
    dag,
    deploy_canary_task,
    eval_qa_task,
    fine_tune_sft_task,
    ingest_task,
)


def test_dag_structure() -> None:
    """Ensure all expected tasks exist with proper dependencies."""

    task_ids = {t.task_id for t in dag.tasks}
    assert task_ids == {
        "ingest",
        "build_dataset",
        "fine_tune_SFT",
        "eval_QA",
        "deploy_canary",
    }
    assert dag.get_task("build_dataset").upstream_task_ids == {"ingest"}
    assert dag.get_task("fine_tune_SFT").upstream_task_ids == {"build_dataset"}
    assert dag.get_task("eval_QA").upstream_task_ids == {"fine_tune_SFT"}
    assert dag.get_task("deploy_canary").upstream_task_ids == {"fine_tune_SFT"}


def test_ingest_pushes_tenant() -> None:
    """The ingest task should push the tenant to XCom."""

    class TI(SimpleNamespace):
        def xcom_push(self, key, value):
            self.store = {key: value}

    ti = TI()
    ingest_task(ti=ti, dag_run=SimpleNamespace(conf={"tenant": "alpha"}))
    assert ti.store["tenant"] == "alpha"


def test_downstream_tasks_propagate_tenant() -> None:
    """Each downstream task should pull the tenant from XCom and re-push it."""

    class TI(SimpleNamespace):
        store = {"tenant": "beta"}

        def xcom_pull(self, key, task_ids=None):
            assert key == "tenant"
            return self.store.get(key)

        def xcom_push(self, key, value):
            self.store[key] = value

    ti = TI()
    for task in (
        build_dataset_task,
        fine_tune_sft_task,
        eval_qa_task,
        deploy_canary_task,
    ):
        task(ti=ti)
        assert ti.store["tenant"] == "beta"
