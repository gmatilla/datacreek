"""Airflow DAG orchestrating Datacreek fine-tuning pipeline.

This DAG defines a multi-step workflow for fine-tuning a model for a
particular tenant. Each task retrieves the tenant identifier from XCom,
allowing downstream steps to operate in a tenant‑scoped context.

The graph of tasks matches the specification:

    ingest  >>   build_dataset   >>  fine_tune_SFT
                                     >>  eval_QA
                                     >>  deploy_canary
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable, Optional

try:  # pragma: no cover - Airflow may not be installed
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    AIRFLOW_AVAILABLE = True
except Exception:  # pragma: no cover - lightweight fallback for tests
    AIRFLOW_AVAILABLE = False

    class DAG:  # type: ignore[misc]
        """Minimal DAG stand‑in with context manager behaviour."""

        def __init__(self, dag_id: str, **_: Any) -> None:
            self.dag_id = dag_id
            self.tasks: list[PythonOperator] = []

        def __enter__(self) -> "DAG":  # pragma: no cover - trivial
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
            return None

        def add_task(self, task: "PythonOperator") -> None:
            self.tasks.append(task)

        def get_task(self, task_id: str) -> "PythonOperator":
            for t in self.tasks:
                if t.task_id == task_id:
                    return t
            raise KeyError(task_id)

    class PythonOperator:  # type: ignore[misc]
        """Very small subset of Airflow's PythonOperator used in tests."""

        def __init__(
            self,
            *,
            task_id: str,
            python_callable: Callable[..., Any],
            dag: Optional[DAG] = None,
        ) -> None:
            self.task_id = task_id
            self.python_callable = python_callable
            self.upstream_task_ids: set[str] = set()
            if dag is not None:
                dag.add_task(self)

        def set_upstream(self, task: "PythonOperator") -> None:
            self.upstream_task_ids.add(task.task_id)

        def __rshift__(self, other: "PythonOperator") -> "PythonOperator":
            other.set_upstream(self)
            return other


def _get_tenant_from_context(**context: Any) -> str:
    """Extract tenant from DAG run configuration.

    Parameters
    ----------
    context: Any
        Execution context provided by Airflow.

    Returns
    -------
    str
        Tenant identifier; defaults to ``"default"`` if missing.
    """

    dag_run = context.get("dag_run") or SimpleNamespace(conf={})
    return dag_run.conf.get("tenant", "default")


def ingest_task(**context: Any) -> None:
    """Starting step: pull tenant from configuration and push to XCom."""

    tenant = _get_tenant_from_context(**context)
    ti = context["ti"]
    ti.xcom_push(key="tenant", value=tenant)


def build_dataset_task(**context: Any) -> None:
    """Stub for dataset build; retrieves tenant from XCom."""

    ti = context["ti"]
    tenant = ti.xcom_pull(key="tenant", task_ids="ingest")
    # Placeholder for dataset building logic per tenant
    ti.xcom_push(key="tenant", value=tenant)


def fine_tune_sft_task(**context: Any) -> None:
    """Stub for supervised fine-tuning step."""

    ti = context["ti"]
    tenant = ti.xcom_pull(key="tenant", task_ids="ingest")
    # Placeholder for fine-tuning logic
    ti.xcom_push(key="tenant", value=tenant)


def eval_qa_task(**context: Any) -> None:
    """Stub for evaluation task."""

    ti = context["ti"]
    tenant = ti.xcom_pull(key="tenant", task_ids="ingest")
    # Placeholder for evaluation logic
    ti.xcom_push(key="tenant", value=tenant)


def deploy_canary_task(**context: Any) -> None:
    """Stub for canary deployment."""

    ti = context["ti"]
    tenant = ti.xcom_pull(key="tenant", task_ids="ingest")
    # Placeholder for deployment logic
    ti.xcom_push(key="tenant", value=tenant)


if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="datacreek_finetune",
        start_date=datetime(2024, 1, 1),
        schedule_interval=None,
        catchup=False,
        doc_md=__doc__,
    ) as dag:
        ingest = PythonOperator(task_id="ingest", python_callable=ingest_task)
        build_dataset = PythonOperator(
            task_id="build_dataset", python_callable=build_dataset_task
        )
        fine_tune_sft = PythonOperator(
            task_id="fine_tune_SFT", python_callable=fine_tune_sft_task
        )
        eval_qa = PythonOperator(task_id="eval_QA", python_callable=eval_qa_task)
        deploy_canary = PythonOperator(
            task_id="deploy_canary", python_callable=deploy_canary_task
        )

        ingest >> build_dataset >> fine_tune_sft
        fine_tune_sft >> [eval_qa, deploy_canary]
else:  # pragma: no cover - executed in lightweight test mode
    dag = DAG(dag_id="datacreek_finetune")
    ingest = PythonOperator(task_id="ingest", python_callable=ingest_task, dag=dag)
    build_dataset = PythonOperator(
        task_id="build_dataset", python_callable=build_dataset_task, dag=dag
    )
    fine_tune_sft = PythonOperator(
        task_id="fine_tune_SFT", python_callable=fine_tune_sft_task, dag=dag
    )
    eval_qa = PythonOperator(task_id="eval_QA", python_callable=eval_qa_task, dag=dag)
    deploy_canary = PythonOperator(
        task_id="deploy_canary", python_callable=deploy_canary_task, dag=dag
    )

    ingest >> build_dataset >> fine_tune_sft
    fine_tune_sft >> eval_qa
    fine_tune_sft >> deploy_canary

__all__ = ["dag"]
