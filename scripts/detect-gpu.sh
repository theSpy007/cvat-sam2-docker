#!/usr/bin/env bash
# =============================================================
# detect-gpu.sh — Detect whether Docker GPU access works
# =============================================================
# Exits 0 if GPU is usable from Docker, 1 otherwise.
# Outputs "cuda" or "cpu" to stdout.
# =============================================================

set -euo pipefail

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
    if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 \
        nvidia-smi &>/dev/null 2>&1; then
        echo "cuda"
        exit 0
    fi
fi

echo "cpu"
exit 1
