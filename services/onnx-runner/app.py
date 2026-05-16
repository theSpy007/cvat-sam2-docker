# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Yannick Otten
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
    polygon: Optional[List[float]] = None  # flat [x1,y1,x2,y2,...] contour points


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


def postprocess_yolo_seg(
    output0: np.ndarray,
    output1: np.ndarray,
    cfg: ModelConfig,
    orig_w: int,
    orig_h: int,
    conf_thresh: float,
    mask_thresh: float,
) -> List[DetectionResult]:
    """
    YOLO instance segmentation postprocessor (Ultralytics end2end export).

    output0: [1, N, 6+32]  — N detections: [x1,y1,x2,y2, conf, cls_id, m0..m31]
    output1: [1, 32, Ph, Pw] — prototype masks (typically 160×160)

    Each instance mask = sigmoid(mask_coeffs @ protos), cropped to bbox,
    resized to original image dimensions.
    """
    results: List[DetectionResult] = []
    labels = cfg.labels or []

    dets   = output0[0]   # [N, 38]
    protos = output1[0]   # [32, Ph, Pw]
    num_proto, ph, pw = protos.shape

    model_h = cfg.input_shape[-2]
    model_w = cfg.input_shape[-1]
    sx = orig_w / model_w
    sy = orig_h / model_h

    for i in range(dets.shape[0]):
        row = dets[i]
        x1, y1, x2, y2, conf = row[0], row[1], row[2], row[3], row[4]
        cls_id      = int(round(float(row[5])))
        mask_coeffs = row[6:]  # 32 values

        if float(conf) < conf_thresh:
            continue

        label_name = labels[cls_id].name if cls_id < len(labels) else str(cls_id)

        # Reconstruct instance mask: [Ph, Pw]
        mask_logits = (mask_coeffs @ protos.reshape(num_proto, -1)).reshape(ph, pw)
        mask_prob   = 1.0 / (1.0 + np.exp(-mask_logits))  # sigmoid
        mask        = (mask_prob > mask_thresh).astype(np.uint8)

        # Crop to bbox region in prototype space to prevent bleed across objects
        px1 = max(0, int(x1 * pw / model_w))
        py1 = max(0, int(y1 * ph / model_h))
        px2 = min(pw, int(x2 * pw / model_w))
        py2 = min(ph, int(y2 * ph / model_h))
        crop = np.zeros_like(mask)
        crop[py1:py2, px1:px2] = mask[py1:py2, px1:px2]

        if not crop.any():
            continue

        # Resize to original image dimensions
        mask_img   = Image.fromarray(crop * 255, mode="L").resize((orig_w, orig_h), Image.BILINEAR)
        mask_final = (np.array(mask_img) > 127).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(mask_final * 255, mode="L").save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode()

        # Extract polygon contour from the full-res mask
        import cv2
        contours, _ = cv2.findContours(
            mask_final * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
        )
        polygon: Optional[List[float]] = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            eps     = 0.005 * cv2.arcLength(largest, True)
            approx  = cv2.approxPolyDP(largest, eps, True)
            if len(approx) >= 3:
                polygon = [coord for pt in approx for coord in (float(pt[0][0]), float(pt[0][1]))]

        results.append(DetectionResult(
            label=label_name,
            label_id=cls_id,
            confidence=float(conf),
            bbox=[float(x1) * sx, float(y1) * sy, float(x2) * sx, float(y2) * sy],
            mask=mask_b64,
            polygon=polygon,
        ))

    return results


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
                if len(class_scores) == 1:
                    # End-to-end NMS export: last column is the class index, not a score vector.
                    # e.g. Ultralytics ONNX with end2end=True → [x1,y1,x2,y2,conf,class_id]
                    cls_id = int(round(float(class_scores[0])))
                    score = float(obj_conf)
                elif len(class_scores) > 1:
                    cls_id = int(class_scores.argmax())
                    score = float(obj_conf * class_scores[cls_id])
                else:
                    cls_id = 0
                    score = float(obj_conf)
                if score < conf_thresh:
                    continue
                label_name = labels[cls_id].name if cls_id < len(labels) else str(cls_id)
                # Scale from model input space (e.g. 640×640) to original image size
                model_h = cfg.input_shape[-2]
                model_w = cfg.input_shape[-1]
                sx = orig_w / model_w
                sy = orig_h / model_h
                results.append(DetectionResult(
                    label=label_name,
                    label_id=cls_id,
                    confidence=score,
                    bbox=[
                        float(x1) * sx, float(y1) * sy,
                        float(x2) * sx, float(y2) * sy,
                    ],
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
    if cfg.postprocessing.output_type == "yolo_seg":
        # Fetch all outputs (bbox+coeffs in [0], prototypes in [1])
        outputs = session.run(None, {cfg.input_name: inp})
        elapsed_ms = (time.perf_counter() - start) * 1000
        detections = postprocess_yolo_seg(
            outputs[0], outputs[1], cfg, orig_w, orig_h, conf_thresh, mask_thresh
        )
    else:
        outputs = session.run([cfg.output_name], {cfg.input_name: inp})
        elapsed_ms = (time.perf_counter() - start) * 1000
        detections = postprocess_output(outputs[0], cfg, orig_w, orig_h, conf_thresh, mask_thresh)

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
