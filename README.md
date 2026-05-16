# cvat-sam2

A self-contained, Docker-only local CVAT stack with SAM2-powered assisted annotation and custom ONNX model auto-annotation. **One command to start everything.**

## What this repository does

| Feature | Details |
|---------|---------|
| **CVAT** | Full CVAT annotation server (open-source) |
| **SAM2 assisted annotation** | Segment Anything 2 for interactive segmentation |
| **ONNX auto-annotation** | Custom segmentation/detection models via ONNX Runtime |
| **GPU acceleration** | Automatic CUDA detection; CPU fallback |
| **Single command** | `./cvat-sam2 up` starts everything |
| **Docker only** | No Python/PyTorch/CUDA installs on the host |

---

## Requirements

| Requirement | Notes |
|-------------|-------|
| **git** | To clone this repository |
| **Docker ≥ 24.0** | With Compose V2 plugin (`docker compose`) |
| **NVIDIA Container Toolkit** | _Optional_ — only for GPU acceleration |

> **No Python, PyTorch, or CUDA installation** is needed on the host.
> Everything runs inside Docker containers.

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/your-org/cvat-sam2.git
cd cvat-sam2

# 2. (Optional) review and edit configuration
cp .env.example .env
nano .env   # change passwords, ports, SAM2 model size, etc.

# 3. Check your system
./cvat-sam2 doctor

# 4. Start the stack
./cvat-sam2 up
```

CVAT will be available at **http://localhost:8080** (or whichever port you set in `.env`).

---

## Commands

```bash
./cvat-sam2 up                          # Build and start everything
./cvat-sam2 down                        # Stop and remove containers
./cvat-sam2 restart                     # Stop then start
./cvat-sam2 status                      # Container status + GPU info
./cvat-sam2 logs                        # Tail all logs
./cvat-sam2 logs sam2                   # Tail SAM2 service logs
./cvat-sam2 doctor                      # System health check
./cvat-sam2 clean                       # Remove containers + ALL data (destructive)
./cvat-sam2 models list                 # List registered ONNX models
./cvat-sam2 models validate             # Validate model.yaml configs
./cvat-sam2 annotate --task 42 --model yolo26m             # Auto-annotate (bboxes)
./cvat-sam2 annotate --task 42 --model yolo26x-seg         # Auto-annotate (polygons)
./cvat-sam2 annotate --task 42 --model yolo26m --overwrite # Replace existing annotations
./cvat-sam2 help                        # Full help
```

---

## GPU setup

GPU acceleration is **automatic** when configured correctly.

1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
2. Run `./cvat-sam2 doctor` — it will verify Docker GPU access.
3. Run `./cvat-sam2 up` — GPU is used automatically if available.

If no GPU is found, the stack starts in **CPU mode** with a warning. SAM2 on CPU is functional but significantly slower.

See [docs/gpu.md](docs/gpu.md) for detailed GPU setup instructions.

---

## CPU fallback

If GPU is unavailable (or NVIDIA Container Toolkit is not configured):

- The stack starts automatically in CPU mode.
- SAM2 inference will be slower (seconds per prediction vs. <1s on GPU).
- All other functionality is identical.

---

## Opening CVAT

After `./cvat-sam2 up`:

1. Open **http://localhost:8080** in your browser.
2. Log in with the credentials from your `.env` file.
   - Default: `admin@example.com` / `ChangeMeNow123!`
3. Create a project and task, upload images.

---

## SAM2 assisted annotation

SAM2 (Segment Anything 2) enables **interactive segmentation** — click a point on an object and get an instant mask.

### How to use

1. Start the stack: `./cvat-sam2 up`
2. Open a CVAT task and click **Open Job**.
3. In the left toolbar, click the **AI Tools** icon (wand icon).
4. Select the **Interaction** tab.
5. Click **Add interaction** → select or configure the SAM2 endpoint:
   - URL: `http://localhost:8000/predict`
   - Or use the internal Docker URL: `http://sam2:8000/predict`
6. Click on objects in the image — SAM2 returns a segmentation mask.
7. Accept, refine, or reject the suggestion.

> **Note:** In the open-source CVAT, adding a custom interaction server requires
> configuring the server URL in the AI Tools panel. See [docs/architecture.md](docs/architecture.md)
> for how the SAM2 service integrates with CVAT.

---

## Custom ONNX segmentation models

Add your own segmentation or detection model in 3 steps:

### Step 1 — Create a model directory

```text
models/
  my-model/
    model.yaml   ← required config
    model.onnx   ← your ONNX weights (not committed to git)
```

### Step 2 — Write model.yaml

Copy `models/example-segmentation/model.yaml` and edit. Key fields depend on model type:

**Detection model (e.g. YOLO with bbox output):**
```yaml
name: my-detector
version: "1.0"
task_type: detection
input_name: images
input_shape: [1, 3, 640, 640]
output_name: output0
preprocessing:
  resize: [640, 640]
  normalize_mean: null   # YOLO: no mean/std, only pixel_scale
  normalize_std: null
  pixel_scale: 255.0
postprocessing:
  output_type: bbox
labels:
  - {id: 0, name: person}
  - {id: 1, name: bicycle}
```

**Instance segmentation model (e.g. YOLO-seg):**
```yaml
name: my-seg-model
version: "1.0"
task_type: segmentation
input_name: images
input_shape: [1, 3, 640, 640]
output_name: output0          # bbox+coeffs tensor; prototype tensor fetched automatically
preprocessing:
  resize: [640, 640]
  normalize_mean: null
  normalize_std: null
  pixel_scale: 255.0
postprocessing:
  output_type: yolo_seg       # reconstructs masks from coefficients × prototypes
  mask_threshold: 0.5
labels:
  - {id: 0, name: person}
  - {id: 1, name: bicycle}
```

See [docs/models.md](docs/models.md) for the full schema reference.

---

## Auto-annotate a dataset/task

```bash
./cvat-sam2 annotate --task <CVAT-task-id> --model <model-name> [--overwrite]
```

Examples:
```bash
# Detection model → uploads bounding box annotations
./cvat-sam2 annotate --task 42 --model yolo26m

# Instance segmentation model → uploads polygon annotations
./cvat-sam2 annotate --task 42 --model yolo26x-seg

# Replace all existing annotations before uploading
./cvat-sam2 annotate --task 42 --model yolo26x-seg --overwrite
```

This will:
1. Authenticate with CVAT.
2. Download each frame.
3. Send it to the ONNX runner for inference.
4. Upload **polygon** annotations (segmentation models) or **bounding box** annotations (detection models) back to CVAT.

> Only labels that exist in the CVAT task are uploaded — detections for unknown classes are skipped.

After running, open the task in CVAT to review and edit the auto-generated annotations.

---

## Configuration

All settings are in `.env` (copied from `.env.example` on first start).

| Variable | Default | Description |
|----------|---------|-------------|
| `CVAT_VERSION` | `v2.20.0` | CVAT Docker image tag |
| `CVAT_PORT` | `8080` | Host port for CVAT |
| `CVAT_SUPERUSER_EMAIL` | `admin@example.com` | Admin email |
| `CVAT_SUPERUSER_PASS` | `ChangeMeNow123!` | Admin password |
| `SAM2_MODEL_ID` | `large` | SAM2 size: `tiny`, `small`, `base_plus`, `large` |
| `SAM2_DEVICE` | `auto` | `auto`, `cuda`, `cpu` |
| `SAM2_PORT` | `8000` | Host port for SAM2 service |
| `ONNX_PORT` | `8001` | Host port for ONNX runner |
| `ONNX_DEVICE` | `auto` | `auto`, `cuda`, `cpu` |
| `POSTGRES_PASSWORD` | _(change me)_ | Database password |

---

## Troubleshooting

See [docs/troubleshooting.md](docs/troubleshooting.md) for common issues.

Quick checks:
```bash
./cvat-sam2 doctor          # full system check
./cvat-sam2 logs            # check all logs
./cvat-sam2 logs sam2       # SAM2 specific logs
```

---

## Cleanup

**Stop the stack (keeps data):**
```bash
./cvat-sam2 down
```

**Remove everything including all data (irreversible):**
```bash
./cvat-sam2 clean
```

---

## Platform notes

| Platform | Status | Notes |
|----------|--------|-------|
| **Linux** | ✅ Fully supported | Primary target |
| **macOS** | ⚠️ CPU only | Docker Desktop has no GPU passthrough for NVIDIA |
| **Windows WSL2** | ⚠️ Partial | GPU may work via WSL2 CUDA; test with `doctor` |

---

## Architecture

See [docs/architecture.md](docs/architecture.md) for a full system diagram.

---

## License

Copyright 2025 Yannick Otten

Licensed under the **Apache License, Version 2.0**. See [LICENSE](LICENSE) for the full text.

### Third-party licenses

| Component | License |
|-----------|---------|
| [CVAT](https://github.com/cvat-ai/cvat) | MIT |
| [SAM2 model weights](https://huggingface.co/facebook/sam2-hiera-large) | Apache 2.0 |
| [HuggingFace Transformers](https://github.com/huggingface/transformers) | Apache 2.0 |
| [PyTorch](https://pytorch.org) | BSD-style |
| [FastAPI](https://fastapi.tiangolo.com) | MIT |
| [OpenCV](https://opencv.org) | Apache 2.0 |
| [NumPy](https://numpy.org) | BSD 3-Clause |
| [Pillow](https://python-pillow.org) | HPND (permissive) |
| [Open Policy Agent](https://www.openpolicyagent.org) | Apache 2.0 |
| [PostgreSQL](https://www.postgresql.org) | PostgreSQL License (permissive) |
| [Redis](https://redis.io) | BSD 3-Clause |
| [nginx](https://nginx.org) | BSD 2-Clause |
