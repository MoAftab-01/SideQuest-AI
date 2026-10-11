# SideQuest AI -- container image.
#
# Deliberately CPU-only. A 4B vision model needs a GPU to be interactive, and
# Render's free tier has none, so this image ships the application and its
# deterministic fallback. Set HF_TOKEN on the service and it will pull Gemma and
# run it on CPU: correct, but slow, and the UI will show the real latency rather
# than pretending otherwise.
#
# The torch install is CPU-only on purpose -- the CUDA wheels add several GB and
# would not be used here.

FROM python:3.11-slim

# Cache-bust marker: forces Render to rebuild from scratch so PYTHONPATH env
# (needed for `from app.models.schemas import ...`) is not served from a
# cached layer that predates the fix. Bump this comment on each redeploy.
# BUILD-CACHE-BUST-2026-10-11

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/backend \
    SIDEQUEST_WEB_DIR=/app/web \
    SIDEQUEST_DATA_DIR=/app/data

WORKDIR /app

# Build tools are needed only for wheels with no manylinux build; keeping them in
# a separate layer keeps them out of the final image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

# Torch first, CPU index explicitly. Without this the default index pulls the
# CUDA build and the image balloons past 5 GB for no benefit on this host.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch>=2.4

COPY pyproject.toml ./
COPY backend ./backend
COPY web ./web
COPY eval ./eval
COPY scripts ./scripts

RUN pip install --no-deps -e . \
 && pip install fastapi "uvicorn[standard]" pydantic python-multipart pillow numpy transformers accelerate safetensors sentencepiece

# Player progress is written here. On Render this is an ephemeral filesystem:
# progress persists for the life of the instance and is lost on redeploy.
# That is a deliberate trade-off -- see the privacy section of the README --
# and is reported honestly rather than described as durable storage.
RUN mkdir -p /app/data

EXPOSE 8080

# Report unhealthy until the app answers, so a cold container is not declared
# healthy before it can serve.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8080/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
