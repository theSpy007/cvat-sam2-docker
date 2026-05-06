"""
ONNX Runner Service
===================
FastAPI HTTP service that runs custom ONNX segmentation models.

Endpoints:
  GET  /health                     — Liveness probe
  GET  /models                     — List registered models
  GET  /models/{name}              — Get model config
  POST /models/{name}/predict      — Run inference on an image
  POST /models/{name}/annotate     — Alias for /predict (CVAT naming)

The service scans ONNX_MODELS_DIR (mounted volume) for model directories
each containing model.yaml and optionally model.onnx.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import time
from typing import List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from registry import registry
from schema import ModelConfig

# ──────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[onnx-runner] %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("onnx-runner")

PORT = int(os.environ.get("PORT", 8001))

# ──────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ONNX Model Runner",
    description="Custom ONNX segmentation model inference for CVAT auto-annotation",
    version="1.0.0",
)

# ──────────────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────────────
class InferenceRequest(BaseModel):
    image: str  # base64-encoded image
    confidence_threshold: Optional[float] = None
    mask_threshold: Optional[float] = None


class DetectionResult(BaseModel):
    label: str
    label_id: int
    confidence: float
    bbox: List[float]             # [x1, y1, x2, y2] in pixel coords
    mask: Optional[str] = None    # base64 PNG mask if segmentation


class InferenceResponse(BaseModel):
    model: str
    detections: List[DetectionResult]
    inference_ms: float

# ──────────────────────────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────────────────────────
def decode_image(b64: str) -> Image.Image:
    try:
        data = base64.b64decode(b64)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


def preprocess_image(image: Image.Image, cfg: ModelConfig) -> np.ndarray:
    """Resize, normalize, and format image per model config."""
    pre = cfg.preprocessing
    target_h, target_w = (
        (pre.resize[0], pre.resize[1])
        if pre.resize
        else (cfg.input_shape[-2], cfg.input_shape[-1])
    )

    image = image.resize((target_w, target_h), Image.BILINEAR)

    if pre.channel_order == "BGR":
        image = image.convert("RGB")
        arr = np.array(image)[:, :, ::-1]  # RGB → BGR
    else:
        arr = np.array(image)

    arr = arr.astype(np.float32) / pre.pixel_scale

    if pre.normalize_mean and pre.normalize_std:
        mean = np.array(pre.normalize_mean, dtype=np.float32)
        std  = np.array(pre.normalize_std,  dtype=np.float32)
        arr  = (arr - mean) / std

    # HWC → NCHW or NHWC
    if pre.layout == "NCHW":
        arr = arr.transpose(2, 0, 1)   # HWC → CHW
        arr = arr[np.newaxis, ...]     # CHW → NCHW
    else:
        arr = arr[np.newaxis, ...]     # HWC → NHWC

    if cfg.input_dtype == "float16":
        arr = arr.astype(np.float16)
    elif cfg.input_dtype == "uint8":
        arr = arr.astype(np.uint8)

    return arr


def postprocess_output(
    output: np.ndarray,
    cfg: ModelConfig,
    orig_w: int,
    orig_h: int,
    conf_thresh: float,
    mask_thresh: float,
) -> List[DetectionResult]:
    """
    Generic postprocessor. Handles:
      - Binary mask output: shape [1, H, W] or [1, 1, H, W]
      - Multi-class segmentation: shape [1, C, H, W]
      - Detection (YOLO-style): shape [1, N, 5+C]
    """
    results: List[DetectionResult] = []
    post = cfg.postprocessing
    labels = cfg.labels or []

    out = output.squeeze()  # remove batch dim

    if post.output_type == "mask":
        # ── Segmentation mask output ─────────────────────────────
        if post.sigmoid:
            out = 1.0 / (1.0 + np.exp(-out))
        if post.softmax and out.ndim == 3:
            exp_o = np.exp(out - out.max(axis=0))
            out = exp_o / exp_o.sum(axis=0)

        if out.ndim == 2:
            # Binary mask
            mask = (out > mask_thresh).astype(np.uint8)
            if mask.any():
                mask_img = Image.fromarray(mask * 255, mode="L").resize(
                    (orig_w, orig_h), Image.NEAREST
                )
                buf = io.BytesIO()
                mask_img.save(buf, format="PNG")
                mask_b64 = base64.b64encode(buf.getvalue()).decode()
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                rmin, rmax = int(np.where(rows)[0][[0, -1]])
                cmin, cmax = int(np.where(cols)[0][[0, -1]])
                label_name = labels[0].name if labels else "object"
                results.append(DetectionResult(
                    label=label_name,
                    label_id=0,
                    confidence=float(out[mask.astype(bool)].mean()),
                    bbox=[
                        cmin * orig_w / mask.shape[1],
                        rmin * orig_h / mask.shape[0],
                        cmax * orig_w / mask.shape[1],
                        rmax * orig_h / mask.shape[0],
                    ],
                    mask=mask_b64,
                ))
        elif out.ndim == 3:
            # Multi-class: C x H x W — pick argmax class per pixel
            class_map = out.argmax(axis=0)
            for cls_id in np.unique(class_map):
                if cls_id == 0:
                    continue  # skip background
                mask = (class_map == cls_id).astype(np.uint8)
                if not mask.any():
                    continue
                score = float(out[cls_id][mask.astype(bool)].mean())
                if score < conf_thresh:
                    continue
                mask_img = Image.fromarray(mask * 255, mode="L").resize(
                    (orig_w, orig_h), Image.NEAREST
                )
                buf = io.BytesIO()
                mask_img.save(buf, format="PNG")
                mask_b64 = base64.b64encode(buf.getvalue()).decode()
                label_name = labels[int(cls_id)].name if int(cls_id) < len(labels) else str(cls_id)
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                rmin, rmax = int(np.where(rows)[0][[0, -1]])
                cmin, cmax = int(np.where(cols)[0][[0, -1]])
                results.append(DetectionResult(
                    label=label_name,
                    label_id=int(cls_id),
                    confidence=score,
                    bbox=[
                        cmin * orig_w / mask.shape[1],
                        rmin * orig_h / mask.shape[0],
                        cmax * orig_w / mask.shape[1],
                        rmax * orig_h / mask.shape[0],
                    ],
                    mask=mask_b64,
                ))

    elif post.output_type == "bbox":
        # ── YOLO-style bbox output [N, 5+C] ─────────────────────
        if out.ndim == 2:
            for i in range(out.shape[0]):
                row = out[i]
                if len(row) < 5:
                    continue
                x1, y1, x2, y2, obj_conf = row[:5]
                class_scores = row[5:]
                if len(class_scores) > 0:
                    cls_id = int(class_scores.argmax())
                    score = float(obj_conf * class_scores[cls_id])
                else:
                    cls_id = 0
                    score = float(obj_conf)
                if score < conf_thresh:
                    continue
                label_name = labels[cls_id].name if cls_id < len(labels) else str(cls_id)
                results.append(DetectionResult(
                    label=label_name,
                    label_id=cls_id,
                    confidence=score,
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                ))

    return results

# ──────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    models = registry.list_models()
    return {"status": "ok", "models_registered": len(models)}


@app.get("/models")
async def list_models():
    return registry.list_models()


@app.get("/models/{model_name}")
async def get_model(model_name: str):
    cfg = registry.get_config(model_name)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    return cfg.model_dump()


@app.post("/models/{model_name}/predict")
@app.post("/models/{model_name}/annotate")
async def predict(model_name: str, req: InferenceRequest):
    """Run ONNX inference with a base64-encoded image."""
    try:
        session, cfg = registry.get_session(model_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    image = decode_image(req.image)
    orig_w, orig_h = image.size

    conf_thresh = req.confidence_threshold or cfg.confidence_threshold
    mask_thresh = req.mask_threshold or cfg.postprocessing.mask_threshold

    # Preprocess
    inp = preprocess_image(image, cfg)

    # Run ONNX inference
    start = time.perf_counter()
    outputs = session.run([cfg.output_name], {cfg.input_name: inp})
    elapsed_ms = (time.perf_counter() - start) * 1000

    output = outputs[0]

    # Postprocess
    detections = postprocess_output(output, cfg, orig_w, orig_h, conf_thresh, mask_thresh)

    return InferenceResponse(
        model=model_name,
        detections=detections,
        inference_ms=round(elapsed_ms, 2),
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
    )
