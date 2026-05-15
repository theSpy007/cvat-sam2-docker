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

# ──────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Nuclio Proxy for CVAT + SAM2")

# ── Nuclio API endpoints ───────────────────────────────────────────

@app.get("/api/functions")
async def list_functions():
    """Return a dict of functions (Nuclio format)."""
    return {"sam2": FUNCTION_DESCRIPTOR}


@app.get("/api/functions/{name}")
async def get_function(name: str):
    if name != "sam2":
        raise HTTPException(status_code=404, detail=f"function {name!r} not found")
    return FUNCTION_DESCRIPTOR


# ── CVAT dashboard invocation endpoint ────────────────────────────

@app.post("/api/function_invocations")
async def invoke_via_dashboard(request: Request):
    """
    CVAT sends a POST /api/function_invocations when INVOKE_METHOD=dashboard.
    The function name is passed in the x-nuclio-function-name header.
    This endpoint handles all function invocations.
    """
    function_name = request.headers.get("x-nuclio-function-name", "")
    if function_name != "sam2":
        raise HTTPException(status_code=404, detail=f"function {function_name!r} not found")
    return await _handle_invoke(request)


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


# ── Helpers ───────────────────────────────────────────────────────

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
