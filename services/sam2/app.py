"""
SAM2 Assisted Annotation Service
=================================
A FastAPI HTTP service that wraps Segment Anything 2 (SAM2) for
use with CVAT's assisted annotation workflow.

Endpoints:
  GET  /health          — Liveness probe
  GET  /info            — Model/device information
  POST /predict         — Run SAM2 on an image + prompts
  POST /segment         — Alias for /predict (CVAT convention)

The service downloads the SAM2 model on first start and caches it
in the volume-mounted directory (SAM2_MODEL_CACHE).

Device selection (SAM2_DEVICE env var):
  auto  — use CUDA if available, else CPU
  cuda  — force CUDA (fails if unavailable)
  cpu   — force CPU
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[sam2] %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("sam2")

# ──────────────────────────────────────────────────────────────────
# Configuration (env vars)
# ──────────────────────────────────────────────────────────────────
MODEL_ID_MAP = {
    "tiny":      "facebook/sam2-hiera-tiny",
    "small":     "facebook/sam2-hiera-small",
    "base_plus": "facebook/sam2-hiera-base-plus",
    "large":     "facebook/sam2-hiera-large",
}

RAW_MODEL_ID   = os.environ.get("SAM2_MODEL_ID", "large").lower()
HF_MODEL_ID    = MODEL_ID_MAP.get(RAW_MODEL_ID, RAW_MODEL_ID)
DEVICE_CFG     = os.environ.get("SAM2_DEVICE", "auto").lower()
MODEL_CACHE    = os.environ.get("SAM2_MODEL_CACHE", "/models/sam2_cache")
PORT           = int(os.environ.get("PORT", 8000))

# ──────────────────────────────────────────────────────────────────
# Device resolution
# ──────────────────────────────────────────────────────────────────
def resolve_device() -> str:
    import torch  # lazy import — torch ships with the base image
    if DEVICE_CFG == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    elif DEVICE_CFG == "cuda":
        if not torch.cuda.is_available():
            log.error("SAM2_DEVICE=cuda requested but CUDA is not available!")
            dev = "cpu"
        else:
            dev = "cuda"
    else:
        dev = "cpu"
    log.info("Using device: %s  (SAM2_DEVICE=%s)", dev, DEVICE_CFG)
    return dev

# ──────────────────────────────────────────────────────────────────
# Model loading (lazy, cached globally)
# ──────────────────────────────────────────────────────────────────
_processor = None
_model      = None
_device     = None

def load_model():
    """Load the SAM2 model via HuggingFace transformers (first call only)."""
    global _processor, _model, _device
    if _model is not None:
        return

    log.info("Loading SAM2 model: %s ...", HF_MODEL_ID)
    start = time.time()

    try:
        # AutoProcessor resolves to Sam2VideoProcessor for SAM2 models.
        # Sam2VideoProcessor correctly handles input_points/labels/boxes;
        # Sam2ImageProcessorFast does NOT (it silently drops them with an
        # "Unused or unrecognized kwargs" warning, causing every click to
        # produce a default center mask).
        from transformers import AutoProcessor, Sam2Model
        import torch

        _device = resolve_device()

        cache_dir = Path(MODEL_CACHE)
        cache_dir.mkdir(parents=True, exist_ok=True)

        _processor = AutoProcessor.from_pretrained(
            HF_MODEL_ID,
            cache_dir=str(cache_dir),
        )
        _model = Sam2Model.from_pretrained(
            HF_MODEL_ID,
            cache_dir=str(cache_dir),
        ).to(_device)
        _model.eval()

        elapsed = time.time() - start
        log.info("SAM2 model loaded in %.1fs on %s", elapsed, _device)

    except Exception as exc:
        log.error("Failed to load SAM2 model: %s", exc)
        raise

# ──────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SAM2 Annotation Service",
    description="Assisted annotation using Segment Anything 2",
    version="1.0.0",
)

# ──────────────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────────────
class Point(BaseModel):
    x: float
    y: float
    label: int = Field(1, description="1=foreground, 0=background")


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class PredictRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image (JPEG or PNG)")
    points: Optional[List[Point]] = None
    boxes:  Optional[List[BBox]] = None


class MaskResult(BaseModel):
    mask: str   # base64-encoded PNG mask
    score: float
    bbox: List[float]  # [x1, y1, x2, y2]


class PredictResponse(BaseModel):
    masks: List[MaskResult]
    model_id: str
    device: str

# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────
def decode_image(b64: str) -> Image.Image:
    try:
        data = base64.b64decode(b64)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


def encode_mask(mask_array: np.ndarray) -> str:
    """Convert a boolean H×W mask to a base64-encoded PNG."""
    img = Image.fromarray((mask_array * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def mask_to_bbox(mask: np.ndarray) -> List[float]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return [0.0, 0.0, 0.0, 0.0]
    row_idx = np.where(rows)[0]
    col_idx = np.where(cols)[0]
    rmin, rmax = int(row_idx[0]), int(row_idx[-1])
    cmin, cmax = int(col_idx[0]), int(col_idx[-1])
    return [float(cmin), float(rmin), float(cmax), float(rmax)]

# ──────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Pre-load the model on startup so the first request is fast."""
    try:
        load_model()
    except Exception as exc:
        log.error("Startup model load failed: %s", exc)
        # Don't crash the server — health check will report unhealthy


@app.get("/health")
async def health():
    if _model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "loading", "message": "Model not yet loaded"},
        )
    return {"status": "ok", "model": HF_MODEL_ID, "device": _device}


@app.get("/info")
async def info():
    return {
        "model_id":      HF_MODEL_ID,
        "device":        _device or DEVICE_CFG,
        "model_loaded":  _model is not None,
        "cache_dir":     MODEL_CACHE,
    }


@app.post("/predict", response_model=PredictResponse)
@app.post("/segment", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    Run SAM2 segmentation with optional point / box prompts.

    If no prompts are given, SAM2 runs in automatic mask generation mode
    on the full image (returns top-N masks).
    """
    load_model()  # no-op if already loaded

    import torch

    image = decode_image(req.image)
    img_w, img_h = image.size

    results: List[MaskResult] = []

    with torch.no_grad():
        if req.points or req.boxes:
            # ── Prompted segmentation ───────────────────────────────
            input_points, input_labels, input_boxes = None, None, None

            if req.points:
                # Sam2VideoProcessor expects:
                #   input_points: depth-4 [image][object][point][x,y]
                #   input_labels: depth-3 [image][object][point_label]
                input_points = [[[[p.x, p.y] for p in req.points]]]
                input_labels = [[[p.label for p in req.points]]]

            if req.boxes:
                # input_boxes: depth-3 [image][box][x1,y1,x2,y2]
                input_boxes = [[[b.x1, b.y1, b.x2, b.y2] for b in req.boxes]]

            inputs = _processor(
                images=image,
                input_points=input_points,
                input_labels=input_labels,
                input_boxes=input_boxes,
                return_tensors="pt",
            ).to(_device)

            outputs = _model(**inputs)

            # transformers ≥4.47: post_process_masks(masks, original_sizes)
            # returns a list of tensors (one per batch item), binarized to bool.
            # Scores come from outputs.iou_scores, not post_process_masks.
            processed_masks = _processor.post_process_masks(
                outputs.pred_masks,
                inputs["original_sizes"],
            )

            # processed_masks[0]: (point_batch_size, num_masks, H, W) bool
            # iou_scores[0]:      (point_batch_size, num_masks)
            batch_masks = processed_masks[0]   # (pb, N, H, W)
            batch_scores = outputs.iou_scores[0]  # (pb, N)

            for pb in range(batch_masks.shape[0]):
                for mi in range(batch_masks.shape[1]):
                    m = batch_masks[pb, mi].cpu().numpy()
                    s = float(batch_scores[pb, mi].cpu().item())
                    results.append(MaskResult(
                        mask=encode_mask(m),
                        score=s,
                        bbox=mask_to_bbox(m),
                    ))

        else:
            # ── Automatic mask generation ────────────────────────────
            # Use the model's grid-based automatic mode
            inputs = _processor(images=image, return_tensors="pt").to(_device)
            outputs = _model(**inputs)

            processed_masks = _processor.post_process_masks(
                outputs.pred_masks,
                inputs["original_sizes"],
            )

            # processed_masks[0]: (point_batch_size, num_masks, H, W) bool
            batch_masks = processed_masks[0]
            batch_scores = outputs.iou_scores[0]  # (pb, N)

            # Flatten across point_batch_size, return top-5 by score
            flat_masks = batch_masks.reshape(-1, *batch_masks.shape[-2:])
            flat_scores = batch_scores.reshape(-1).cpu().numpy()
            top_n = min(5, flat_masks.shape[0])
            top_idx = np.argsort(flat_scores)[::-1][:top_n]
            for i in top_idx:
                m = flat_masks[i].cpu().numpy()
                s = float(flat_scores[i])
                results.append(MaskResult(
                    mask=encode_mask(m),
                    score=s,
                    bbox=mask_to_bbox(m),
                ))

    return PredictResponse(
        masks=results,
        model_id=HF_MODEL_ID,
        device=_device,
    )


# ──────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info",
        access_log=True,
    )
