# GPU Setup Guide

## Overview

CVAT-SAM2 uses NVIDIA GPU acceleration for:
- **SAM2** inference (segmentation prediction)
- **ONNX Runner** inference (custom model prediction)

GPU support is **optional** — the stack runs on CPU if no GPU is available.

---

## Requirements

| Component | Requirement |
|-----------|------------|
| NVIDIA GPU | Any CUDA-capable GPU (Compute Capability ≥ 6.0) |
| NVIDIA Driver | ≥ 520 recommended |
| NVIDIA Container Toolkit | Must be installed and configured |
| Docker | ≥ 24.0 with `nvidia` runtime registered |

---

## Installation — NVIDIA Container Toolkit

### Ubuntu / Debian

```bash
# Add NVIDIA package repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Verification

```bash
# Verify NVIDIA driver
nvidia-smi

# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

Both commands should show your GPU. If only the second fails, reinstall nvidia-container-toolkit.

---

## How GPU is selected automatically

When you run `./cvat-sam2 up`, the script:

1. Checks `nvidia-smi` — detects host GPU.
2. Runs a Docker GPU probe container — verifies Container Toolkit.
3. If both pass: applies `docker-compose.gpu.yml` (sets `runtime: nvidia`, GPU reservations, `SAM2_DEVICE=cuda`).
4. If either fails: applies `docker-compose.cpu.yml` (`SAM2_DEVICE=cpu`) and prints a warning.

You can verify which mode is active:
```bash
./cvat-sam2 status
```

---

## Manual GPU override

If automatic detection is wrong, force a mode via `.env`:

```bash
# Force GPU
SAM2_DEVICE=cuda
ONNX_DEVICE=cuda

# Force CPU
SAM2_DEVICE=cpu
ONNX_DEVICE=cpu
```

---

## Performance comparison

| Mode | SAM2 prediction time (typical) |
|------|-------------------------------|
| GPU (RTX 3080) | ~0.1–0.3s per click |
| GPU (RTX 4090) | <0.1s per click |
| CPU (8-core, 32GB RAM) | ~3–10s per click |

Using SAM2 on CPU is usable but noticeably slower for interactive annotation.

---

## ONNX Runtime GPU

The ONNX runner uses `CUDAExecutionProvider` when GPU is detected.

> **Note:** The ONNX runner Docker image ships with `onnxruntime` (CPU). If you need
> explicit GPU ONNX acceleration, rebuild with:
>
> ```dockerfile
> RUN pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
> ```
>
> Or adjust `services/onnx-runner/requirements.txt` before running `./cvat-sam2 up`.

---

## Troubleshooting GPU

### "Docker cannot access GPU" after installing toolkit

```bash
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### "CUDA out of memory"

Reduce SAM2 model size in `.env`:
```
SAM2_MODEL_ID=small   # or: tiny
```

### "nvidia runtime not found"

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### WSL2 / Windows

GPU in WSL2 works for recent Windows 11 + WSL2 + NVIDIA Game Ready / Studio drivers.
Install the Container Toolkit **inside WSL2**, not on Windows host:
```bash
# Inside WSL2 Ubuntu shell
sudo apt-get install -y nvidia-container-toolkit
```

---

## macOS

NVIDIA GPU passthrough is **not supported** on macOS with Docker Desktop.
The stack runs in CPU mode automatically.
