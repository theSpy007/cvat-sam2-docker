# Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                         │
│                   (project: cvat-sam2)                         │
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  cvat_proxy  │    │  cvat_server │    │   cvat_ui        │  │
│  │  (nginx)     │───▶│  (Django)    │◀───│   (React/nginx)  │  │
│  │  :8080       │    │  :8080       │    │                  │  │
│  └─────────────┘    └──────────────┘    └──────────────────┘  │
│                             │                                   │
│                    ┌────────┴─────────┐                        │
│                    │                  │                         │
│              ┌─────┴──────┐  ┌───────┴──────┐                 │
│              │  cvat_db    │  │cvat_redis_*  │                 │
│              │ (Postgres)  │  │              │                 │
│              └─────────────┘  └──────────────┘                │
│                                                                 │
│  ┌──────────────────────┐  ┌────────────────────────────────┐ │
│  │  sam2                │  │  onnx_runner                   │ │
│  │  (FastAPI + SAM2)    │  │  (FastAPI + ONNXRuntime)       │ │
│  │  :8000               │  │  :8001                         │ │
│  │                      │  │  mounts: ./models/             │ │
│  │  GPU: cuda           │  │  GPU: cuda or CPU              │ │
│  │  CPU: fallback       │  │                                │ │
│  └──────────────────────┘  └────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Services

### cvat_proxy (Nginx)
- Reverse proxy that exposes CVAT on `localhost:CVAT_PORT`.
- Routes `/api/*` to the Django server.
- Routes everything else to the React UI.

### cvat_server (Django)
- The main CVAT annotation server.
- Handles authentication, task/project management, annotation storage.
- Connects to PostgreSQL for persistent storage and Redis for caching/queuing.

### cvat_ui (React)
- The CVAT web frontend.
- Served by Nginx inside the container.

### cvat_db (PostgreSQL 15)
- Persistent annotation database.
- Volume: `cvat_db_data`

### cvat_redis_inmem / cvat_redis_ondisk
- In-memory cache and on-disk queue for CVAT workers.

### cvat_worker_general / cvat_worker_annotation
- Background workers handling CVAT tasks (export, import, annotation jobs).

### sam2 (SAM2 Service)
- **FastAPI** application wrapping Segment Anything 2.
- Downloads the SAM2 model from Hugging Face on first start.
- Model cached in Docker volume `sam2_model_cache`.
- GPU: `SAM2_DEVICE=cuda` when NVIDIA GPU is available.
- CPU: fallback with `SAM2_DEVICE=cpu`.
- Exposed at `http://localhost:SAM2_PORT` on the host.

### onnx_runner (ONNX Model Runner)
- **FastAPI** application running custom ONNX segmentation models.
- Scans `./models/` for `model.yaml` configs.
- Loads ONNX sessions with `CUDAExecutionProvider` when GPU available.
- Exposed at `http://localhost:ONNX_PORT` on the host.

## Compose file structure

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Base stack definition |
| `docker-compose.gpu.yml` | GPU resource overrides (applied automatically) |
| `docker-compose.cpu.yml` | CPU-explicit overrides (applied when no GPU) |
| `docker-compose.serverless.yml` | Nuclio serverless platform (optional) |

## SAM2 CVAT Integration

CVAT's open-source edition supports custom **interactive annotation providers** 
through its AI Tools panel. The SAM2 service exposes a REST API at `/predict`
that accepts:

- A base64-encoded image
- Optional point prompts (foreground/background)
- Optional bounding box hints

The API returns binary segmentation masks that CVAT renders as polygon/mask annotations.

**Integration path:**
1. User opens CVAT AI Tools → Interaction tab.
2. Configures endpoint: `http://sam2:8000/predict` (internal) or `http://localhost:8000/predict`.
3. Clicks on image → request sent to SAM2 service → mask returned → displayed in CVAT.

## GPU detection

The `./cvat-sam2` CLI script runs two checks:

1. `nvidia-smi` — checks host NVIDIA driver.
2. `docker run --gpus all nvidia/cuda:... nvidia-smi` — verifies Docker GPU access.

Only if **both** pass is the GPU compose override applied.

## Data persistence

| Volume | Contents |
|--------|---------|
| `cvat_db_data` | PostgreSQL data |
| `cvat_redis_data` | Redis on-disk data |
| `cvat_data` | CVAT uploaded images, annotations |
| `cvat_keys` | CVAT secret keys |
| `cvat_logs` | CVAT log files |
| `sam2_model_cache` | Downloaded SAM2 model weights |
| `./models/` (bind mount) | User ONNX models (host directory) |
