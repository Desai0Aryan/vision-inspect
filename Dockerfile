# syntax=docker/dockerfile:1.7
# VisionInspect: multi-stage image.
#   docker build -t visioninspect .                  -> "runtime": slim, ONNX Runtime INT8/FP32 + classical (no PyTorch)
#   docker build --target full -t visioninspect:full .  -> adds PyTorch + Ultralytics (PyTorch backend, training, export)

# ---------- 1. build the React dashboard ----------
FROM node:22-alpine AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---------- 2. slim runtime ----------
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VI_JOBS_DIR=/data/jobs \
    VI_FRONTEND_DIR=/app/frontend/dist \
    VI_MODELS_DIR=/app/models \
    VI_DEMO_VIDEO=/app/data/demo/conveyor.mp4
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-runtime.txt ./
RUN pip install -r requirements-runtime.txt
COPY visioninspect/ visioninspect/
COPY api/ api/
COPY scripts/make_demo_video.py scripts/
COPY configs/ configs/
COPY models/*.onnx models/
COPY outputs/ outputs/
COPY data/demo/conveyor.mp4 data/demo/conveyor.gt.json data/demo/
COPY --from=ui /ui/dist frontend/dist
COPY docker/entrypoint.sh /usr/local/bin/entrypoint
RUN chmod +x /usr/local/bin/entrypoint \
 && useradd --create-home --uid 10001 app \
 && mkdir -p /data/jobs && chown -R app /data /app
USER app
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=4 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"
ENTRYPOINT ["entrypoint"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------- 3. full: PyTorch backend, training and export ----------
FROM runtime AS full
USER root
COPY requirements-train.txt ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
      torch torchvision "ultralytics>=8.3" "onnx>=1.16" onnxslim
COPY models/yolo_pretrained.pt models/
COPY scripts/ scripts/
RUN chown -R app /app
USER app

# The default build target is the last stage, so make the slim one the default explicitly:
FROM runtime AS default
