# =============================================================
# SAM2 Service Dockerfile
# =============================================================
# Builds a FastAPI HTTP service that exposes SAM2 (Segment
# Anything 2) for interactive/assisted annotation.
#
# The base image uses CUDA 12.1 when building for GPU.
# At runtime SAM2_DEVICE=auto selects cuda/cpu automatically.
# =============================================================

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

LABEL maintainer="cvat-sam2"
LABEL description="SAM2 assisted annotation service for CVAT"

# Prevent tzdata interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Model cache directory (mounted as a volume)
ENV SAM2_MODEL_CACHE=/models/sam2_cache
RUN mkdir -p /models/sam2_cache

# Default runtime settings
ENV SAM2_MODEL_ID=large
ENV SAM2_DEVICE=auto
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=5 \
    CMD curl -fs http://localhost:8000/health || exit 1

CMD ["python", "app.py"]
