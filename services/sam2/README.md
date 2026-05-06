# SAM2 Service

This directory contains the SAM2 (Segment Anything 2) assisted annotation service.

## What it does

Provides a FastAPI HTTP service that runs SAM2 inference.
CVAT users can use the SAM2 service for **interactive (assisted) annotation** —
clicking points on an object to generate segmentation masks.

## Endpoints

| Method | Path       | Description                            |
|--------|------------|----------------------------------------|
| GET    | `/health`  | Liveness probe                         |
| GET    | `/info`    | Model and device information           |
| POST   | `/predict` | Run SAM2 with point/box prompts        |
| POST   | `/segment` | Alias for `/predict`                   |

## Environment variables

| Variable         | Default                  | Notes                          |
|------------------|--------------------------|--------------------------------|
| `SAM2_MODEL_ID`  | `large`                  | `tiny`, `small`, `base_plus`, `large` |
| `SAM2_DEVICE`    | `auto`                   | `auto`, `cuda`, `cpu`          |
| `SAM2_MODEL_CACHE` | `/models/sam2_cache`   | Volume-mounted model cache dir |
| `PORT`           | `8000`                   | HTTP port                      |

## Model download

The SAM2 model is downloaded from Hugging Face on first start and cached in the
volume `sam2_model_cache`. Subsequent starts reuse the cache.

Models available:
- `tiny`      → `facebook/sam2-hiera-tiny`
- `small`     → `facebook/sam2-hiera-small`
- `base_plus` → `facebook/sam2-hiera-base-plus`
- `large`     → `facebook/sam2-hiera-large` (default, best quality)

## How to use with CVAT

1. Start the stack with `./cvat-sam2 up`.
2. Open CVAT at `http://localhost:8080`.
3. Open a task and start annotating.
4. In the annotation interface, use the **AI Tools** tab → **Interaction** section.
5. Configure a custom endpoint pointing to `http://sam2:8000/predict`.

See [docs/architecture.md](../../docs/architecture.md) for integration details.
