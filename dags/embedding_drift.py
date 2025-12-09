"""Airflow DAG monitoring embedding drift and triggering retraining.

The workflow measures the current drift score and conditionally initiates
retraining of the model when the drift exceeds the critical threshold defined
in :mod:`datacreek.drift`.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any, Optional

from datacreek.drift import should_trigger_retrain

try:  # pragma: no cover - Airflow may not be installed
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    AIRFLOW_AVAILABLE = True
except Exception:  # pragma: no cover - lightweight fallbacks for tests
    AIRFLOW_AVAILABLE = False

    class DAG:  # type: ignore[misc]
        def __init__(self, dag_id: str, **_: Any) -> None:
            self.dag_id = dag_id
            self.tasks: list[PythonOperator] = []

        def __enter__(self) -> "DAG":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def add_task(self, task: "PythonOperator") -> None:
            self.tasks.append(task)

        def get_task(self, task_id: str) -> "PythonOperator":
            for t in self.tasks:
                if t.task_id == task_id:
                    return t
            raise KeyError(task_id)

    class PythonOperator:  # type: ignore[misc]
        def __init__(
            self,
            *,
            task_id: str,
            python_callable: Any,
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


def compute_drift_task(**context: Any) -> None:
    """Push the provided drift value to XCom for downstream use."""

    dag_run = context.get("dag_run") or SimpleNamespace(conf={})
    drift = float(dag_run.conf.get("drift", 0.0))
    context["ti"].xcom_push(key="drift", value=drift)


def trigger_retrain_task(**context: Any) -> None:
    """Trigger retraining if the measured drift is critical.

    The task pulls the drift score from XCom and sets a ``triggered`` flag to
    communicate whether retraining would be scheduled in a full Airflow
    environment.  Using a flag keeps tests lightweight without invoking Airflow
    specific operators.
    """

    ti = context["ti"]
    drift = float(ti.xcom_pull(key="drift", task_ids="compute_drift") or 0.0)
    triggered = should_trigger_retrain(drift)
    ti.xcom_push(key="triggered", value=triggered)


if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="embedding_drift",
        start_date=datetime(2024, 1, 1),
        schedule_interval=None,
        catchup=False,
        doc_md=__doc__,
    ) as dag:
        compute_drift = PythonOperator(
            task_id="compute_drift", python_callable=compute_drift_task
        )
        trigger_retrain = PythonOperator(
            task_id="trigger_retrain", python_callable=trigger_retrain_task
        )
        compute_drift >> trigger_retrain
else:  # pragma: no cover - executed in lightweight test mode
    dag = DAG(dag_id="embedding_drift")
    compute_drift = PythonOperator(
        task_id="compute_drift", python_callable=compute_drift_task, dag=dag
    )
    trigger_retrain = PythonOperator(
        task_id="trigger_retrain", python_callable=trigger_retrain_task, dag=dag
    )
    compute_drift >> trigger_retrain

__all__ = ["dag", "compute_drift_task", "trigger_retrain_task"]
