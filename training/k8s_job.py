"""Kubernetes job specification helpers for training workloads.

These utilities build minimal ``batch/v1`` Job manifests that are aware of
preemptible (spot) GPU nodes.  When ``use_spot`` is True the job tolerates the
``preemptible=true:NoSchedule`` taint so it may schedule onto spot GPU nodes.
Jobs always use ``restartPolicy: OnFailure`` so that the cluster-autoscaler can
reschedule them when a spot node is evicted.
"""

from __future__ import annotations

from typing import Dict, List


def build_training_job(
    name: str, image: str, *, use_spot: bool = True
) -> Dict[str, object]:
    """Return a minimal Kubernetes Job manifest.

    Parameters
    ----------
    name:
        Name of the job and primary container.
    image:
        Container image to execute.
    use_spot:
        If True the job tolerates the ``preemptible`` taint so that it can run
        on spot GPU nodes.  When False no tolerations are applied and the job
        targets on-demand nodes only.

    Returns
    -------
    dict
        Dictionary representing the Job manifest, suitable for serialisation to
        YAML or submission via the Kubernetes API.
    """
    tolerations: List[Dict[str, str]] = []
    if use_spot:
        # Allow scheduling onto nodes tainted as preemptible/spot GPUs.
        tolerations.append(
            {
                "key": "preemptible",
                "operator": "Equal",
                "value": "true",
                "effect": "NoSchedule",
            }
        )

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name},
        "spec": {
            "template": {
                "spec": {
                    "containers": [{"name": name, "image": image}],
                    "restartPolicy": "OnFailure",
                    **({"tolerations": tolerations} if tolerations else {}),
                }
            }
        },
    }


__all__ = ["build_training_job"]
