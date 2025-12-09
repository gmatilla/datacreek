#!/bin/bash
# Install faiss with GPU support if CUDA is available, otherwise use the CPU build.
set -euo pipefail
if command -v nvidia-smi >/dev/null 2>&1; then
    pip install faiss-gpu
else
    pip install faiss-cpu
fi
