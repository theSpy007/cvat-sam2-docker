"""
schema.py — Model configuration schema for ONNX runner.
Validates model.yaml files placed in models/<name>/model.yaml.
"""

from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class PreprocessingConfig(BaseModel):
    """Image preprocessing parameters."""
    resize: Optional[List[int]] = None        # [height, width]
    normalize_mean: Optional[List[float]] = None   # e.g. [0.485, 0.456, 0.406]
    normalize_std:  Optional[List[float]] = None   # e.g. [0.229, 0.224, 0.225]
    pixel_scale:    float = 255.0                  # divide pixel values by this
    channel_order:  Literal["RGB", "BGR"] = "RGB"
    layout:         Literal["NCHW", "NHWC"] = "NCHW"


class PostprocessingConfig(BaseModel):
    """Model output postprocessing parameters."""
    output_type: Literal["mask", "bbox", "polygon"] = "mask"
    mask_threshold: float = Field(0.5, ge=0.0, le=1.0)
    sigmoid: bool = False    # apply sigmoid before threshold
    softmax: bool = False    # apply softmax before threshold


class LabelConfig(BaseModel):
    id:    int
    name:  str
    color: Optional[str] = None  # hex color e.g. "#FF0000"


class ModelConfig(BaseModel):
    """
    Schema for models/<name>/model.yaml

    All fields except name and task_type have reasonable defaults.
    """
    name:          str
    version:       str = "1.0"
    description:   Optional[str] = None
    task_type:     Literal["segmentation", "detection", "classification"] = "segmentation"

    # ONNX I/O
    input_name:   str = "input"
    input_shape:  List[int] = Field(default=[1, 3, 640, 640])  # NCHW
    input_dtype:  Literal["float32", "float16", "uint8"] = "float32"
    output_name:  str = "output"

    # Processing
    preprocessing:          PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    postprocessing:         PostprocessingConfig = Field(default_factory=PostprocessingConfig)
    confidence_threshold:   float = Field(0.5, ge=0.0, le=1.0)
    nms_threshold:          float = Field(0.4, ge=0.0, le=1.0)

    # Labels
    labels: Optional[List[LabelConfig]] = None

    @field_validator("input_shape")
    @classmethod
    def validate_input_shape(cls, v: List[int]) -> List[int]:
        if len(v) not in (3, 4):
            raise ValueError("input_shape must have 3 or 4 dimensions, e.g. [1,3,640,640]")
        return v
