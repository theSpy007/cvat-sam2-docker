# Troubleshooting Guide

Start with the built-in doctor:
```bash
./cvat-sam2 doctor
```

---

## Docker / startup issues

### "Docker daemon is not running"

```bash
sudo systemctl start docker
# Or for rootless Docker:
systemctl --user start docker
```

### "Permission denied" on Docker socket

```bash
sudo usermod -aG docker $USER
# Log out and back in, then try again
```

### "Docker Compose not found"

Install the Compose V2 plugin:
```bash
# Ubuntu/Debian
sudo apt-get install docker-compose-plugin

# Or manually:
DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
mkdir -p $DOCKER_CONFIG/cli-plugins
curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 -o $DOCKER_CONFIG/cli-plugins/docker-compose
chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose
```

---

## CVAT not reachable

### Check container status

```bash
./cvat-sam2 status
./cvat-sam2 logs cvat_server
```

### Common causes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Port 8080 already in use | Another service on that port | Change `CVAT_PORT` in `.env` |
| `cvat_server` not healthy | DB not ready yet | Wait 2 minutes; check `./cvat-sam2 logs cvat_db` |
| `cvat_db` not healthy | DB init failed | Run `./cvat-sam2 clean` then `./cvat-sam2 up` |

### Database initialization failure

```bash
./cvat-sam2 logs cvat_db
# If corrupted:
./cvat-sam2 clean
./cvat-sam2 up
```

---

## SAM2 issues

### SAM2 model download takes too long

The first start downloads the SAM2 model (~2.4GB for `large`). This is normal.
Watch progress:
```bash
./cvat-sam2 logs sam2
```

Use a smaller model to reduce download time:
```bash
# In .env:
SAM2_MODEL_ID=small   # or: tiny (~200MB)
./cvat-sam2 restart
```

### SAM2 health check fails

```bash
./cvat-sam2 logs sam2
# Common causes:
# - Model still downloading (wait, it takes time on first run)
# - Out of GPU memory (reduce SAM2_MODEL_ID or set SAM2_DEVICE=cpu)
# - Network issue downloading model from Hugging Face
```

### SAM2 is very slow (CPU mode)

SAM2 on CPU is 10–30× slower than GPU. Options:
1. Set up GPU (see [docs/gpu.md](gpu.md))
2. Use `SAM2_MODEL_ID=tiny` for faster CPU inference

### SAM2 "CUDA out of memory"

```bash
# In .env:
SAM2_MODEL_ID=small   # free ~1.5GB VRAM vs large
./cvat-sam2 restart
```

---

## ONNX runner issues

### Model not found

```bash
./cvat-sam2 models list
# Check:
# - models/<name>/model.yaml exists
# - model directory name matches what you pass to --model
```

### Model validation fails

```bash
./cvat-sam2 models validate
# Read the error messages — most common issues:
# - Wrong tensor names (find with: python -c "import onnxruntime as ort; s=ort.InferenceSession('model.onnx'); print([i.name for i in s.get_inputs()])")
# - Invalid YAML syntax in model.yaml
# - input_shape has wrong number of dimensions
```

### Inference returns no detections

- Check `confidence_threshold` — lower it to `0.1` to see if anything is returned.
- Verify `input_name` and `output_name` match your model.
- Ensure `preprocessing` matches what your model expects (mean/std normalization).
- Check `postprocessing.sigmoid` — binary models often need `sigmoid: true`.

---

## GPU issues

See [docs/gpu.md](gpu.md) for full GPU setup.

### GPU detected but not used

```bash
./cvat-sam2 doctor
# If "Docker GPU access" fails:
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Wrong GPU mode shown in status

```bash
./cvat-sam2 down
./cvat-sam2 up   # re-runs GPU detection
```

---

## Auto-annotation issues

### "CVAT login failed"

- Check credentials in `.env` (CVAT_SUPERUSER_EMAIL, CVAT_SUPERUSER_PASS).
- Ensure CVAT is running: `./cvat-sam2 status`

### "No frames found for task"

- The task must have images/video uploaded before running annotate.
- Check task ID is correct in the CVAT UI.

### Annotations not appearing in CVAT

- Check that the CVAT task has labels matching your model's label names.
- Review annotate output for errors.
- Look at `./cvat-sam2 logs onnx_runner`

---

## Cleaning up and starting fresh

```bash
# Stop and remove all containers + volumes (ALL DATA LOST)
./cvat-sam2 clean

# Start fresh
./cvat-sam2 up
```

---

## Getting help

1. Run `./cvat-sam2 doctor` — covers most common issues.
2. Check `./cvat-sam2 logs` for error messages.
3. Open an issue on GitHub with the output of `./cvat-sam2 doctor`.
