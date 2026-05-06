#!/usr/bin/env python3
"""
validate_models.py — Validate all model.yaml configs in models/ directory.

Usage:
  python3 scripts/validate_models.py models/
  python3 scripts/validate_models.py models/my-model/
"""

from __future__ import annotations

import sys
import os
import traceback
from pathlib import Path

import yaml
from pydantic import ValidationError

# Add onnx-runner service to path for schema import
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "services" / "onnx-runner"))

from schema import ModelConfig  # noqa: E402


def validate_model_dir(model_dir: Path) -> bool:
    """Validate a single model directory. Returns True if valid."""
    config_path = model_dir / "model.yaml"
    name = model_dir.name

    if not config_path.exists():
        print(f"  [SKIP] {name}: no model.yaml found")
        return True  # Not an error — may be a placeholder dir

    print(f"  [CHECK] {name} ...")

    try:
        with config_path.open() as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            print(f"  [FAIL] {name}: model.yaml must be a YAML mapping (dict)")
            return False

        cfg = ModelConfig(**raw)
        print(f"  [PASS] {name} (task={cfg.task_type}, version={cfg.version})")

        # Additional soft checks
        onnx_files = list(model_dir.glob("*.onnx"))
        if onnx_files:
            print(f"         Weight file: {onnx_files[0].name}")
        else:
            print(f"         [WARN] No .onnx weight file — model cannot run inference until added")

        labels = cfg.labels or []
        if not labels:
            print(f"         [WARN] No labels defined in model.yaml")

        return True

    except yaml.YAMLError as exc:
        print(f"  [FAIL] {name}: invalid YAML — {exc}")
        return False
    except ValidationError as exc:
        print(f"  [FAIL] {name}: schema validation errors:")
        for e in exc.errors():
            field = " → ".join(str(x) for x in e["loc"])
            print(f"         {field}: {e['msg']}")
        return False
    except Exception as exc:
        print(f"  [FAIL] {name}: unexpected error — {exc}")
        traceback.print_exc()
        return False


def main() -> int:
    models_root = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "models"

    if not models_root.exists():
        print(f"[validate_models] Directory not found: {models_root}")
        return 1

    print(f"\nValidating models in: {models_root}\n")

    # Find all model directories
    if (models_root / "model.yaml").exists():
        # Single model directory passed
        dirs = [models_root]
    else:
        dirs = sorted([d for d in models_root.iterdir() if d.is_dir()])

    if not dirs:
        print("  No model directories found.")
        return 0

    passed = 0
    failed = 0
    for d in dirs:
        ok = validate_model_dir(d)
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
