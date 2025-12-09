"""Tests for Kubernetes training job manifest helpers."""

from training.k8s_job import build_training_job


def test_spot_job_includes_toleration_and_restart_policy():
    """Spot jobs should tolerate the preemptible taint and restart on failure."""
    job = build_training_job("train", "img", use_spot=True)

    spec = job["spec"]["template"]["spec"]
    assert spec["restartPolicy"] == "OnFailure"
    tol = spec["tolerations"][0]
    assert tol == {
        "key": "preemptible",
        "operator": "Equal",
        "value": "true",
        "effect": "NoSchedule",
    }


def test_on_demand_job_has_no_tolerations():
    """On-demand jobs must not carry spot tolerations."""
    job = build_training_job("train", "img", use_spot=False)
    spec = job["spec"]["template"]["spec"]
    assert "tolerations" not in spec
