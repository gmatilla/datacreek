"""Tests for tenant namespace Helm template.

These tests render the ``charts/tenant-ns.tpl`` template by simple string
substitution and validate that the resulting YAML defines the expected
Kubernetes objects for isolation and quotas.
"""

from pathlib import Path

import yaml


def render(values: dict) -> list:
    """Render the Helm-like template using ``values`` substitutions."""
    template = Path("charts/tenant-ns.tpl").read_text()
    for key, val in values.items():
        template = template.replace(f"{{{{{key}}}}}", str(val))
    return list(yaml.safe_load_all(template))


def test_namespace_quota_and_network_policy():
    """Template should create namespace, quota, limits and network policy."""
    docs = render(
        {
            "id": "acme",
            "cpu": 4,
            "gpu": 1,
            "maxCpu": 2,
            "maxMemory": "4Gi",
            "reqCpu": "500m",
            "reqMemory": "512Mi",
        }
    )

    namespace, quota, limits, netpol = docs

    assert namespace["metadata"]["name"] == "tenant-acme"
    assert quota["spec"]["hard"]["requests.cpu"] == "4"
    assert quota["spec"]["hard"]["requests.nvidia.com/gpu"] == "1"
    lim = limits["spec"]["limits"][0]
    assert lim["max"]["cpu"] == "2"
    assert lim["max"]["memory"] == "4Gi"
    assert "Egress" in netpol["spec"]["policyTypes"]
    ns_selector = netpol["spec"]["egress"][0]["to"][0]["namespaceSelector"][
        "matchLabels"
    ]
    assert ns_selector["kubernetes.io/metadata.name"] == "tenant-acme"


def test_gpu_nodegroups_template():
    """The GPU nodegroup template defines spot and on-demand pools."""
    template = Path("charts/gpu-nodegroups.tpl").read_text()
    docs = list(yaml.safe_load_all(template))
    spot, ondemand = docs

    assert spot["metadata"]["name"] == "spot-gpu"
    taint = spot["spec"]["taints"][0]
    assert taint == {
        "key": "preemptible",
        "value": "true",
        "effect": "NoSchedule",
    }
    assert ondemand["metadata"]["name"] == "on-demand-gpu"
    assert "taints" not in ondemand["spec"]
