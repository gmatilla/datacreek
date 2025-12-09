# Helm template for tenant-specific namespace with quotas and network isolation
# This template instantiates the Kubernetes objects required to provision
# an isolated namespace for a tenant. It sets compute quotas, per-pod
# limits and a NetworkPolicy that prevents egress to other namespaces.
#
# Parameters:
#   id:        tenant identifier appended to the namespace name
#   cpu:       total CPU quota allocated to the namespace
#   gpu:       total GPU quota (`nvidia.com/gpu`) allocated
#   maxCpu:    maximum CPU per container
#   maxMemory: maximum memory per container
#   reqCpu:    default CPU request per container
#   reqMemory: default memory request per container
---
apiVersion: v1
kind: Namespace
metadata:
  name: tenant-{{id}}
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: compute-quota
  namespace: tenant-{{id}}
spec:
  hard:
    requests.cpu: "{{cpu}}"
    requests.nvidia.com/gpu: "{{gpu}}"
---
apiVersion: v1
kind: LimitRange
metadata:
  name: compute-limits
  namespace: tenant-{{id}}
spec:
  limits:
    - type: Container
      max:
        cpu: "{{maxCpu}}"
        memory: "{{maxMemory}}"
      defaultRequest:
        cpu: "{{reqCpu}}"
        memory: "{{reqMemory}}"
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-cross-tenant-egress
  namespace: tenant-{{id}}
spec:
  podSelector: {}
  policyTypes:
    - Egress
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: tenant-{{id}}
