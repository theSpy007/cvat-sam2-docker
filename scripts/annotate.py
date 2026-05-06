#!/usr/bin/env python3
"""
annotate.py — Trigger CVAT task auto-annotation using an ONNX model.

This script iterates over frames in a CVAT task, sends each frame to the
ONNX runner for inference, and uploads the resulting annotations back to CVAT
via the CVAT REST API.

Usage (called by ./cvat-sam2 annotate):
  python3 scripts/annotate.py \
    --task 42 \
    --model example-segmentation \
    --cvat-url http://localhost:8080 \
    --cvat-user admin@example.com \
    --cvat-pass secret \
    --onnx-url http://localhost:8001
"""

from __future__ import annotations

import argparse
import base64
import logging
import sys
from pathlib import Path
from typing import List, Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="[annotate] %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("annotate")

# ──────────────────────────────────────────────────────────────────
# CVAT API helpers
# ──────────────────────────────────────────────────────────────────

class CVATClient:
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._login(username, password)

    def _login(self, username: str, password: str) -> None:
        resp = self.session.post(
            f"{self.base_url}/api/auth/login",
            json={"username": username, "password": password},
            timeout=30,
        )
        if resp.status_code not in (200, 204):
            raise RuntimeError(
                f"CVAT login failed ({resp.status_code}): {resp.text[:500]}"
            )
        log.info("Authenticated with CVAT as %s", username)

    def get_task(self, task_id: int) -> dict:
        resp = self.session.get(
            f"{self.base_url}/api/tasks/{task_id}", timeout=30
        )
        resp.raise_for_status()
        return resp.json()

    def get_frames(self, task_id: int) -> List[dict]:
        """Return a list of frame info dicts for a task."""
        resp = self.session.get(
            f"{self.base_url}/api/tasks/{task_id}/data/meta", timeout=30
        )
        resp.raise_for_status()
        return resp.json().get("frames", [])

    def get_frame_image_b64(self, task_id: int, frame_id: int) -> str:
        """Download a task frame as base64-encoded JPEG."""
        url = f"{self.base_url}/api/tasks/{task_id}/data?type=frame&number={frame_id}&quality=original"
        resp = self.session.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode()

    def upload_annotations(self, task_id: int, annotations: dict) -> None:
        """Upload annotations in CVAT JSON format to a task."""
        resp = self.session.patch(
            f"{self.base_url}/api/tasks/{task_id}/annotations",
            json=annotations,
            timeout=60,
        )
        resp.raise_for_status()
        log.info("Uploaded annotations to task %d", task_id)

    def get_labels(self, task_id: int) -> List[dict]:
        task = self.get_task(task_id)
        return task.get("labels", [])

# ──────────────────────────────────────────────────────────────────
# ONNX runner helpers
# ──────────────────────────────────────────────────────────────────

class ONNXRunnerClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def predict(self, model_name: str, image_b64: str) -> List[dict]:
        resp = requests.post(
            f"{self.base_url}/models/{model_name}/predict",
            json={"image": image_b64},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("detections", [])

# ──────────────────────────────────────────────────────────────────
# Annotation conversion: ONNX detections → CVAT format
# ──────────────────────────────────────────────────────────────────

def build_cvat_annotations(
    detections_per_frame: dict,
    cvat_labels: List[dict],
) -> dict:
    """
    Convert ONNX runner detection results into CVAT annotation format.
    Returns a dict suitable for the CVAT PATCH /api/tasks/{id}/annotations endpoint.
    """
    label_map = {lbl["name"]: lbl["id"] for lbl in cvat_labels}

    shapes = []
    for frame_id, detections in detections_per_frame.items():
        for det in detections:
            label_name = det.get("label", "object")
            label_id = label_map.get(label_name)
            if label_id is None:
                # Use first available label as fallback
                label_id = cvat_labels[0]["id"] if cvat_labels else 0

            bbox = det.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

            shapes.append({
                "type": "rectangle",
                "occluded": False,
                "outside": False,
                "z_order": 0,
                "rotation": 0.0,
                "points": [x1, y1, x2, y2],
                "frame": frame_id,
                "label_id": label_id,
                "group": 0,
                "source": "auto",
                "attributes": [],
            })

    return {
        "version": 0,
        "tags": [],
        "shapes": shapes,
        "tracks": [],
    }


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-annotate a CVAT task via ONNX runner")
    parser.add_argument("--task",       required=True, type=int,  help="CVAT task ID")
    parser.add_argument("--model",      required=True,            help="ONNX model name")
    parser.add_argument("--cvat-url",   required=True,            help="CVAT base URL")
    parser.add_argument("--cvat-user",  required=True,            help="CVAT username/email")
    parser.add_argument("--cvat-pass",  required=True,            help="CVAT password")
    parser.add_argument("--onnx-url",   required=True,            help="ONNX runner base URL")
    parser.add_argument("--frames",     type=int, default=0,
                        help="Max frames to annotate (0 = all)")
    args = parser.parse_args()

    cvat    = CVATClient(args.cvat_url, args.cvat_user, args.cvat_pass)
    onnx_c  = ONNXRunnerClient(args.onnx_url)

    log.info("Target task: %d, model: %s", args.task, args.model)

    # Get task metadata
    task    = cvat.get_task(args.task)
    labels  = cvat.get_labels(args.task)
    frames  = cvat.get_frames(args.task)

    if not frames:
        log.warning("No frames found for task %d", args.task)
        return 0

    max_frames = args.frames if args.frames > 0 else len(frames)
    log.info("Processing %d / %d frames ...", min(max_frames, len(frames)), len(frames))

    detections_per_frame: dict = {}
    for i, frame in enumerate(frames[:max_frames]):
        frame_id = i  # CVAT frame index
        log.info("  Frame %d / %d ...", i + 1, min(max_frames, len(frames)))
        try:
            image_b64 = cvat.get_frame_image_b64(args.task, frame_id)
            dets = onnx_c.predict(args.model, image_b64)
            detections_per_frame[frame_id] = dets
            log.info("    → %d detections", len(dets))
        except Exception as exc:
            log.error("  Frame %d failed: %s", frame_id, exc)
            detections_per_frame[frame_id] = []

    annotations = build_cvat_annotations(detections_per_frame, labels)
    total_shapes = len(annotations["shapes"])
    log.info("Total shapes to upload: %d", total_shapes)

    if total_shapes > 0:
        cvat.upload_annotations(args.task, annotations)
        log.info("Done. Open CVAT task %d to review annotations.", args.task)
    else:
        log.warning("No detections found — no annotations uploaded.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
