# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Yannick Otten
"""
nuclio-proxy — Lightweight Nuclio API emulator for CVAT + SAM2
===============================================================
This service replaces a real Nuclio installation. It:

1. Speaks just enough of the Nuclio API so CVAT can discover
   the SAM2 interactor function  (GET /api/functions[/{name}]).

2. Handles the CVAT interactor invocation protocol on the
   same HTTP port, translating it to our SAM2 FastAPI service
   and returning polygon points that CVAT can render on the canvas.

CVAT → nuclio-proxy (port 8070)
   • GET  /api/functions         → list functions
   • GET  /api/functions/sam2    → inspect function
   • POST /                      → invoke (CVAT direct-mode calls this port)

nuclio-proxy → sam2 (port 8000)
   • POST /predict               → run SAM2 inference

Environment variables:
  SAM2_HOST       Host/IP of the SAM2 service (default: sam2)
  SAM2_PORT       Port of the SAM2 service (default: 8000)
  PROXY_PORT      Port this proxy listens on (default: 8070)
  FUNCTION_HOST_PORT  The host port CVAT will call for invocation.
                  CVAT (inside Docker) uses host.docker.internal:<port>.
                  Set this to the same value as PROXY_PORT unless
                  there is a separate port mapping.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image

# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────
SAM2_HOST = os.environ.get("SAM2_HOST", "sam2")
SAM2_PORT = int(os.environ.get("SAM2_PORT", 8000))
SAM2_BASE = f"http://{SAM2_HOST}:{SAM2_PORT}"
PROXY_PORT = int(os.environ.get("PROXY_PORT", 8070))
# Host port that CVAT will use when calling via host.docker.internal
FUNCTION_HOST_PORT = int(os.environ.get("FUNCTION_HOST_PORT", PROXY_PORT))
# ONNX runner — used to discover and invoke ONNX detector models
ONNX_RUNNER_URL = os.environ.get("ONNX_RUNNER_URL", "http://onnx_runner:8001")

logging.basicConfig(
    level=logging.INFO,
    format="[nuclio-proxy] %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("nuclio-proxy")

# ──────────────────────────────────────────────────────────────────
# The fake Nuclio function descriptor — what CVAT reads to render
# the "magic wand" toolbar button and understand the protocol.
# ──────────────────────────────────────────────────────────────────
FUNCTION_DESCRIPTOR: Dict[str, Any] = {
    "metadata": {
        "name": "sam2",
        "namespace": "nuclio",
        "annotations": {
            # CVAT reads these fields to know how to present / invoke the function
            "name":                   "SAM2 Interactive Segmentation",
            "type":                   "interactor",
            "framework":              "pytorch",
            "spec":                   "[]",       # JSON-encoded label list (empty = any label)
            "min_pos_points":         "1",
            "min_neg_points":         "0",
            "startswith_box":         "false",
            "startswith_box_optional":"false",
            "animated_gif":           "",
            "help_message":           "Click to add foreground points, right-click for background. SAM2 will generate the segment.",
            # CVAT v2.20 requires version >= 2 to enable the Interact button.
            # (disabled:!u||f||u.version<2 in the minified frontend JS)
            "version":                "2",
        },
    },
    "spec": {
        "description": "SAM2 (Segment Anything 2) interactive image segmentation via HuggingFace transformers.",
    },
    "status": {
        # httpPort is the HOST port CVAT calls when INVOKE_METHOD=direct.
        # CVAT (inside Docker on Linux) reaches this via host.docker.internal.
        "httpPort": FUNCTION_HOST_PORT,
        "state": "ready",
    },
}


async def _fetch_onnx_descriptors() -> Dict[str, Dict[str, Any]]:
    """
    Query the ONNX runner's /models endpoint and build a Nuclio-style
    descriptor dict for every detection model that has ONNX weights.
    CVAT reads these descriptors to populate its Detectors tool panel.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{ONNX_RUNNER_URL}/models")
            resp.raise_for_status()
            model_list: Dict[str, dict] = resp.json()
    except Exception as exc:
        log.warning("Could not reach ONNX runner to list models: %s", exc)
        return {}

    descriptors: Dict[str, Dict[str, Any]] = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for model_key, info in model_list.items():
            if not info.get("has_weights", False):
                continue  # skip placeholders without ONNX weights
            # Fetch full config to obtain the labels list for CVAT's spec field
            try:
                cfg_resp = await client.get(f"{ONNX_RUNNER_URL}/models/{model_key}")
                cfg_resp.raise_for_status()
                cfg = cfg_resp.json()
            except Exception as exc:
                log.warning("Could not fetch config for ONNX model %r: %s", model_key, exc)
                cfg = info

            labels = cfg.get("labels") or []
            spec = json.dumps([{"id": lbl["id"], "name": lbl["name"]} for lbl in labels])
            fn_name = f"onnx-{model_key}"
            descriptors[fn_name] = {
                "metadata": {
                    "name": fn_name,
                    "namespace": "nuclio",
                    "annotations": {
                        "name": f"{cfg.get('name', model_key)} (ONNX)",
                        "type": "detector",
                        "framework": "onnx",
                        "spec": spec,
                        "version": "2",
                    },
                },
                "spec": {
                    "description": cfg.get("description") or "",
                },
                "status": {
                    "httpPort": FUNCTION_HOST_PORT,
                    "state": "ready",
                },
            }
    return descriptors


# ──────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Nuclio Proxy for CVAT + SAM2")

# ── Nuclio API endpoints ───────────────────────────────────────────

@app.get("/api/functions")
async def list_functions():
    """Return a dict of all functions: SAM2 interactor + ONNX detectors."""
    fns: Dict[str, Any] = {"sam2": FUNCTION_DESCRIPTOR}
    fns.update(await _fetch_onnx_descriptors())
    return fns


@app.get("/api/functions/{name}")
async def get_function(name: str):
    if name == "sam2":
        return FUNCTION_DESCRIPTOR
    onnx = await _fetch_onnx_descriptors()
    if name in onnx:
        return onnx[name]
    raise HTTPException(status_code=404, detail=f"function {name!r} not found")


# ── CVAT dashboard invocation endpoint ────────────────────────────

@app.post("/api/function_invocations")
async def invoke_via_dashboard(request: Request):
    """
    CVAT sends a POST /api/function_invocations when INVOKE_METHOD=dashboard.
    The function name is passed in the x-nuclio-function-name header.
    This endpoint handles all function invocations.
    """
    function_name = request.headers.get("x-nuclio-function-name", "")
    if function_name == "sam2":
        return await _handle_invoke(request)
    if function_name.startswith("onnx-"):
        model_name = function_name[len("onnx-"):]
        return await _handle_onnx_invoke(request, model_name)
    raise HTTPException(status_code=404, detail=f"function {function_name!r} not found")


# ── CVAT direct invocation endpoint ───────────────────────────────

@app.post("/")
async def invoke_direct(request: Request):
    """
    CVAT sends a POST / when INVOKE_METHOD=direct.
    Same logic — translate CVAT's interactor payload to SAM2 and return polygons.
    """
    return await _handle_invoke(request)


async def _handle_invoke(request: Request):
    """
    Shared handler for both dashboard and direct invocation.
    Payload:
        {
            "image":      "<base64-jpeg>",
            "pos_points": [[x, y], ...],   # foreground clicks
            "neg_points": [[x, y], ...],   # background clicks
            "obj_bbox":   [x1, y1, x2, y2] # optional hint box
        }
    Returns a list of polygon point arrays:
        [[x1,y1, x2,y2, ...], ...]
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    image_b64: str = body.get("image", "")
    pos_points: List[List[float]] = body.get("pos_points", [])
    neg_points: List[List[float]] = body.get("neg_points", [])
    obj_bbox: Optional[List[float]] = body.get("obj_bbox")

    if not image_b64:
        raise HTTPException(status_code=400, detail="Missing 'image' field")

    # Build SAM2 request
    points = [
        {"x": p[0], "y": p[1], "label": 1}
        for p in pos_points
    ] + [
        {"x": p[0], "y": p[1], "label": 0}
        for p in neg_points
    ]

    boxes = []
    if obj_bbox and len(obj_bbox) == 4:
        boxes = [{"x1": obj_bbox[0], "y1": obj_bbox[1],
                  "x2": obj_bbox[2], "y2": obj_bbox[3]}]

    sam2_payload = {
        "image":  image_b64,
        "points": points or None,
        "boxes":  boxes or None,
    }

    log.info(
        "invoke: pos=%d neg=%d bbox=%s",
        len(pos_points), len(neg_points), obj_bbox is not None
    )

    # Forward to SAM2 service
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{SAM2_BASE}/predict", json=sam2_payload)
            resp.raise_for_status()
            sam2_result = resp.json()
    except httpx.HTTPStatusError as exc:
        log.error("SAM2 returned error %s: %s", exc.response.status_code, exc.response.text[:500])
        raise HTTPException(status_code=502, detail=f"SAM2 service error: {exc.response.status_code}")
    except Exception as exc:
        log.error("SAM2 request failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"SAM2 service unreachable: {exc}")

    masks: List[Dict[str, Any]] = sam2_result.get("masks", [])
    if not masks:
        log.info("SAM2 returned no masks")
        return {"mask": [], "points": []}

    # Pick the best mask (highest score)
    best = max(masks, key=lambda m: m.get("score", 0.0))
    mask_b64: str = best["mask"]

    # Convert PNG mask → CVAT interactor response format
    # CVAT frontend expects: {mask: [[bool,...], ...], points: [[x,y], ...]}
    # - mask:   2-D boolean list (H × W) — used for RLE storage as a MASK shape
    # - points: contour vertices as [[x,y], ...] — used for POLYGON shape
    # See renderInteractorBlock / runInteractionRequest in the CVAT JS bundle.
    try:
        result = _mask_to_cvat_response(mask_b64)
    except Exception:
        log.error("Mask conversion failed:\n%s", traceback.format_exc())
        return {"mask": [], "points": []}

    log.info(
        "Returning mask %dx%d with %d contour point(s)",
        len(result["mask"]),
        len(result["mask"][0]) if result["mask"] else 0,
        len(result["points"]),
    )
    return result


async def _handle_onnx_invoke(request: Request, model_name: str) -> Any:
    """
    Handle a CVAT Nuclio detector invocation for an ONNX model.

    CVAT sends:
        {"image": "<base64>", "threshold": 0.5,
         "mapping": {"model_label": {"name": "cvat_label", "attributes": []}}}

    CVAT expects back a list of shape dicts:
        [{"confidence": "0.85", "label": "cvat_label",
          "points": [x1, y1, x2, y2], "type": "rectangle", "attributes": []}]
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    image_b64: str = body.get("image", "")
    threshold: float = float(body.get("threshold", 0.5))
    mapping: Dict[str, Any] = body.get("mapping") or {}

    if not image_b64:
        raise HTTPException(status_code=400, detail="Missing 'image' field")

    log.info("ONNX detect: model=%s threshold=%.2f", model_name, threshold)

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{ONNX_RUNNER_URL}/models/{model_name}/predict",
                json={"image": image_b64, "confidence_threshold": threshold},
            )
            resp.raise_for_status()
            result = resp.json()
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        if code == 404:
            raise HTTPException(status_code=404, detail=f"ONNX model {model_name!r} not found")
        log.error("ONNX runner error %s: %s", code, exc.response.text[:500])
        raise HTTPException(status_code=502, detail=f"ONNX runner error: {code}")
    except Exception as exc:
        log.error("ONNX runner unreachable: %s", exc)
        raise HTTPException(status_code=502, detail=f"ONNX runner unreachable: {exc}")

    cvat_results = []
    for det in result.get("detections", []):
        model_label = det.get("label", "")
        mapped = mapping.get(model_label, {})
        cvat_label = mapped.get("name", model_label) if mapped else model_label

        # 1. Pre-computed polygon (OBB 4-corner or seg contour from runner)
        poly_field = det.get("polygon")
        if poly_field and len(poly_field) >= 6:
            cvat_results.append({
                "confidence": str(round(det.get("confidence", 0.0), 4)),
                "label": cvat_label,
                "points": poly_field,
                "type": "polygon",
                "attributes": [],
            })
            continue

        # 2. Raw mask — convert to polygon contour
        mask_b64_det = det.get("mask")
        if mask_b64_det:
            try:
                poly_points = _mask_to_polygon_points(mask_b64_det)
            except Exception as exc:
                log.warning("mask→polygon failed for %s: %s", model_label, exc)
                poly_points = None
            if poly_points:
                cvat_results.append({
                    "confidence": str(round(det.get("confidence", 0.0), 4)),
                    "label": cvat_label,
                    "points": poly_points,
                    "type": "polygon",
                    "attributes": [],
                })
                continue

        # 3. Detection-only model — use axis-aligned bbox rectangle
        cvat_results.append({
            "confidence": str(round(det.get("confidence", 0.0), 4)),
            "label": cvat_label,
            "points": det.get("bbox", []),
            "type": "rectangle",
            "attributes": [],
        })

    log.info("ONNX detect: model=%s → %d detections", model_name, len(cvat_results))
    return cvat_results


# ── Helpers ───────────────────────────────────────────────────────

def _mask_to_polygon_points(mask_b64: str) -> Optional[List[float]]:
    """
    Convert a base64 PNG mask to a flat CVAT polygon points list [x1,y1,x2,y2,...].
    Returns None if no contour is found (empty/degenerate mask).
    """
    import cv2

    data = base64.b64decode(mask_b64)
    arr  = np.array(Image.open(io.BytesIO(data)).convert("L"), dtype=np.uint8)
    _, binary = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(largest, True)
    approx  = cv2.approxPolyDP(largest, epsilon, True)
    if len(approx) < 3:
        return None
    return [coord for pt in approx for coord in (float(pt[0][0]), float(pt[0][1]))]


def _mask_to_cvat_response(mask_b64: str) -> Dict[str, Any]:
    """
    Convert a base64-encoded grayscale PNG mask to the dict format that
    CVAT's interactor frontend expects:

        {
            "mask":   [[bool, ...], ...],  # 2-D list, H rows × W cols
            "points": [[x, y], ...],       # contour vertices (largest contour)
        }

    The JS handler (runInteractionRequest) uses:
      • mask   → KJe.utils.mask2Rle(n.mask.flat()) for MASK shape storage
      • points → approximateResponsePoints(n.points) for POLYGON shape display
    """
    import cv2

    data = base64.b64decode(mask_b64)
    img = Image.open(io.BytesIO(data)).convert("L")
    arr = np.array(img, dtype=np.uint8)

    # Threshold to binary
    _, binary = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY)

    # 2-D boolean list (H × W) — what CVAT's mask2Rle expects
    mask_2d: List[List[bool]] = (binary > 0).tolist()

    # Find contours for the polygon points
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    if not contours:
        return {"mask": mask_2d, "points": []}

    # Pick the largest contour and approximate it
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)

    # [[x, y], ...] — what CVAT's approxPoly / flat() pipeline expects
    points: List[List[float]] = [
        [float(p[0][0]), float(p[0][1])] for p in approx
    ]

    return {"mask": mask_2d, "points": points}


# ── Health check ──────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Simple health probe."""
    return {"status": "ok"}


# ── Entrypoint ────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=PROXY_PORT,
        log_level="info",
        access_log=True,
    )
