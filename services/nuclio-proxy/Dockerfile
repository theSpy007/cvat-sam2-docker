# ────────────────────────────────────────────────────────────
# Nuclio-proxy — lightweight mock Nuclio API + SAM2 bridge
# ────────────────────────────────────────────────────────────
# Minimal Python image. No GPU needed — all inference happens
# in the SAM2 service; this is just a protocol translator.
FROM python:3.11-slim

WORKDIR /app

# System deps for opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./

EXPOSE 8070

CMD ["python", "app.py"]
