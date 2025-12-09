# NodeGroup definitions for spot and on-demand GPU nodes.
# Rendered statically; cluster-autoscaler and GPU Operator consume these to
# manage separate pools.  The spot pool carries a ``preemptible`` taint so that
# only tolerant jobs schedule onto it.
---
apiVersion: autoscaling/v1
kind: NodeGroup
metadata:
  name: spot-gpu
spec:
  taints:
    - key: preemptible
      value: "true"
      effect: NoSchedule
  labels:
    nvidia.com/gpu: "true"
---
apiVersion: autoscaling/v1
kind: NodeGroup
metadata:
  name: on-demand-gpu
spec:
  labels:
    nvidia.com/gpu: "true"
