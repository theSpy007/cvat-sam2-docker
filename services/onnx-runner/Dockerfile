# =============================================================
# ONNX Runner Dockerfile
# =============================================================
# Runs custom ONNX segmentation models for CVAT auto-annotation.
#
# Build args:
#   ONNX_RUNTIME_PKG — which onnxruntime package to install.
#     CPU (default): onnxruntime>=1.17.0
#     GPU:           onnxruntime-gpu>=1.17.0
#
# The GPU compose override sets ONNX_DEVICE=cuda. The service
# will use CUDAExecutionProvider if onnxruntime-gpu is installed.
# If onnxruntime (CPU) is installed but ONNX_DEVICE=cuda is set,
# the service falls back to CPU with a warning.
# =============================================================

FROM python:3.11-slim

# Declare build arg AFTER FROM (Docker scoping rule)
ARG ONNX_RUNTIME_PKG="onnxruntime>=1.17.0"

LABEL maintainer="cvat-sam2"
LABEL description="ONNX runtime inference service for custom segmentation models"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install base requirements, then the selected onnxruntime package.
# To use GPU: build with --build-arg ONNX_RUNTIME_PKG="onnxruntime-gpu>=1.17.0"
# The ./cvat-sam2 CLI passes this automatically when GPU is detected.
ARG ONNX_RUNTIME_PKG="onnxruntime>=1.17.0"
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "${ONNX_RUNTIME_PKG}"

COPY app.py registry.py schema.py ./

ENV ONNX_DEVICE=auto
ENV ONNX_MODELS_DIR=/models/user
ENV PORT=8001

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
    CMD curl -fs http://localhost:8001/health || exit 1

CMD ["python", "app.py"]
