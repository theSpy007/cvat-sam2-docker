# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Yannick Otten
"""
registry.py — Model registry for the ONNX runner.
Discovers and loads all models from the ONNX_MODELS_DIR directory.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import yaml
from schema import ModelConfig

log = logging.getLogger("onnx-runner.registry")

MODELS_DIR = Path(os.environ.get("ONNX_MODELS_DIR", "/models/user"))


class ModelRegistry:
    """Discovers model.yaml files and loads ONNX sessions on demand."""

    def __init__(self):
        self._configs: Dict[str, ModelConfig] = {}
        self._sessions: Dict[str, object] = {}  # ort.InferenceSession
        self._refresh()

    def _refresh(self) -> None:
        """Scan ONNX_MODELS_DIR for model.yaml files."""
        self._configs.clear()
        if not MODELS_DIR.exists():
            log.warning("Models directory does not exist: %s", MODELS_DIR)
            return

        for model_dir in sorted(MODELS_DIR.iterdir()):
            if not model_dir.is_dir():
                continue
            config_path = model_dir / "model.yaml"
            if not config_path.exists():
                log.debug("Skipping %s — no model.yaml", model_dir.name)
                continue
            try:
                with config_path.open() as f:
                    raw = yaml.safe_load(f)
                cfg = ModelConfig(**raw)
                self._configs[model_dir.name] = cfg
                log.info("Registered model: %s (task=%s)", model_dir.name, cfg.task_type)
            except Exception as exc:
                log.error("Failed to load model config %s: %s", config_path, exc)

    def list_models(self) -> Dict[str, dict]:
        return {
            name: {
                "name": cfg.name,
                "version": cfg.version,
                "task_type": cfg.task_type,
                "description": cfg.description,
                "has_weights": self._has_onnx(name),
            }
            for name, cfg in self._configs.items()
        }

    def _has_onnx(self, model_key: str) -> bool:
        model_dir = MODELS_DIR / model_key
        return bool(list(model_dir.glob("*.onnx")))

    def get_config(self, model_key: str) -> Optional[ModelConfig]:
        if model_key not in self._configs:
            self._refresh()
        return self._configs.get(model_key)

    def get_session(self, model_key: str):
        """Return a cached ONNX InferenceSession, loading if needed."""
        if model_key in self._sessions:
            return self._sessions[model_key], self._configs[model_key]

        cfg = self.get_config(model_key)
        if cfg is None:
            raise KeyError(f"Model '{model_key}' not found in registry")

        model_dir = MODELS_DIR / model_key
        onnx_files = list(model_dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(
                f"No .onnx weights found in {model_dir}. "
                "Add model.onnx to the model directory."
            )

        onnx_path = str(onnx_files[0])
        log.info("Loading ONNX session: %s", onnx_path)

        import onnxruntime as ort

        device = os.environ.get("ONNX_DEVICE", "auto").lower()
        providers = _select_providers(device)
        log.info("ONNX ExecutionProviders: %s", providers)

        session_opts = ort.SessionOptions()
        session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            onnx_path,
            sess_options=session_opts,
            providers=providers,
        )
        self._sessions[model_key] = session
        return session, cfg


def _select_providers(device: str) -> list:
    """Return ONNX Runtime execution providers based on device config."""
    import onnxruntime as ort
    available = ort.get_available_providers()

    if device == "cuda" or (device == "auto" and "CUDAExecutionProvider" in available):
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            log.warning(
                "ONNX_DEVICE=cuda requested but CUDAExecutionProvider is not available. "
                "Falling back to CPU. Install onnxruntime-gpu for GPU support."
            )

    return ["CPUExecutionProvider"]


# Singleton registry
registry = ModelRegistry()
