# Custom ONNX Models Guide

## Overview

The ONNX runner service supports any segmentation or detection model exported to the
ONNX format. Models are loaded at runtime from the `models/` directory — no code
changes required.

---

## Directory layout

```
models/
  <model-name>/
    model.yaml    ← required: model configuration
    model.onnx    ← required for inference (NOT committed to git)
    labels.txt    ← optional: one label per line
    README.md     ← optional: notes
```

---

## model.yaml schema

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | **required** | Display name |
| `version` | string | `"1.0"` | Version string |
| `description` | string | — | Optional description |
| `task_type` | enum | `segmentation` | `segmentation` \| `detection` \| `classification` |
| `input_name` | string | `input` | ONNX input tensor name |
| `input_shape` | list[int] | `[1,3,640,640]` | Input tensor shape (NCHW or NHWC) |
| `input_dtype` | enum | `float32` | `float32` \| `float16` \| `uint8` |
| `output_name` | string | `output` | ONNX output tensor name |
| `preprocessing.resize` | list[int] | from input_shape | `[height, width]` |
| `preprocessing.normalize_mean` | list[float] | — | Per-channel mean |
| `preprocessing.normalize_std` | list[float] | — | Per-channel std |
| `preprocessing.pixel_scale` | float | `255.0` | Divide pixels by this |
| `preprocessing.channel_order` | enum | `RGB` | `RGB` \| `BGR` |
| `preprocessing.layout` | enum | `NCHW` | `NCHW` \| `NHWC` |
| `postprocessing.output_type` | enum | `mask` | `mask` \| `bbox` \| `polygon` |
| `postprocessing.mask_threshold` | float | `0.5` | Mask binarization threshold |
| `postprocessing.sigmoid` | bool | `false` | Apply sigmoid before threshold |
| `postprocessing.softmax` | bool | `false` | Apply softmax (multi-class) |
| `confidence_threshold` | float | `0.5` | Min detection confidence |
| `nms_threshold` | float | `0.4` | NMS IoU threshold (bbox only) |
| `labels` | list | — | List of `{id, name, color}` |

---

## Step-by-step: adding a new model

### 1. Find your model's tensor names

```python
import onnxruntime as ort

s = ort.InferenceSession("model.onnx")
for inp in s.get_inputs():
    print(f"Input:  {inp.name!r}  shape={inp.shape}  dtype={inp.type}")
for out in s.get_outputs():
    print(f"Output: {out.name!r}  shape={out.shape}  dtype={out.type}")
```

### 2. Create the model directory

```bash
mkdir -p models/my-model
cp models/example-segmentation/model.yaml models/my-model/model.yaml
```

### 3. Edit model.yaml

Update at minimum:
- `name`
- `input_name` (from step 1)
- `input_shape` (from step 1)
- `output_name` (from step 1)
- `labels` (your class names)

### 4. Copy your weights

```bash
cp /path/to/your/model.onnx models/my-model/model.onnx
```

> ONNX weights are ignored by `.gitignore` and will not be committed.

### 5. Validate

```bash
./cvat-sam2 models validate
```

### 6. Restart

```bash
./cvat-sam2 restart
```

### 7. Test via API

```bash
# List models
curl http://localhost:8001/models

# Get model config
curl http://localhost:8001/models/my-model

# Run inference (requires base64-encoded image)
python3 - << 'EOF'
import base64, requests

with open("test.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

resp = requests.post(
    "http://localhost:8001/models/my-model/predict",
    json={"image": img_b64},
)
print(resp.json())
EOF
```

---

## Using models for auto-annotation

```bash
./cvat-sam2 annotate --task <CVAT-task-id> --model my-model
```

The annotate command will:
1. Connect to CVAT and fetch all frames.
2. Run each frame through the ONNX model.
3. Upload bounding box / mask annotations to CVAT.

> **Important:** The CVAT task must have **label names that match** your `model.yaml`
> labels. Create the labels in CVAT before running auto-annotation.

---

## Output types

### `mask` — segmentation output

The model output should be:
- Shape `[1, H, W]` — binary mask
- Shape `[1, C, H, W]` — multi-class (argmax per pixel)

With `sigmoid: true` for binary models (logit output).
With `softmax: false` for most multi-class models.

### `bbox` — detection output (YOLO-style)

The model output should be:
- Shape `[1, N, 5+C]` where `N` = detections, `5` = `[x1, y1, x2, y2, obj_conf]`, `C` = class scores

---

## Labels file (alternative to model.yaml labels)

You can use a simple `labels.txt` instead of listing labels in `model.yaml`:

```
background
object-class-1
object-class-2
```

Line index = label ID (0-based).

---

## GPU acceleration for ONNX

By default, the ONNX runner ships with `onnxruntime` (CPU).

For GPU inference, rebuild with `onnxruntime-gpu`:

```bash
# Edit services/onnx-runner/requirements.txt
# Change: onnxruntime>=1.17.0
# To:     onnxruntime-gpu>=1.17.0

./cvat-sam2 restart
```

The runner will automatically use `CUDAExecutionProvider` when a GPU is present.
