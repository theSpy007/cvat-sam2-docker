# Example Segmentation Model

This directory is a placeholder/example model for the CVAT-SAM2 ONNX runner.

## To use your own model

1. **Copy this directory** to `models/<your-model-name>/`
2. **Edit `model.yaml`** to match your model's architecture
3. **Copy your weights** as `model.onnx` into the directory
4. **Validate**: `./cvat-sam2 models validate`
5. **Restart**: `./cvat-sam2 restart`

## Finding your model's tensor names

```bash
python3 - << 'EOF'
import onnxruntime as ort
s = ort.InferenceSession("model.onnx")
print("Inputs: ", [(i.name, i.shape, i.type) for i in s.get_inputs()])
print("Outputs:", [(o.name, o.shape, o.type) for o in s.get_outputs()])
EOF
```

## Directory layout

```
models/
  <model-name>/
    model.yaml    ← required: config/schema
    model.onnx    ← required for inference (NOT committed to git)
    labels.txt    ← optional: one label per line (overrides model.yaml labels)
    README.md     ← optional: model notes
```

See `docs/models.md` for the full schema reference.
