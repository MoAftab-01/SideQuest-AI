# SideQuest AI -- container image.
#
# OpenRouter hosts model inference, so this image needs no local model weights.

FROM python:3.11-slim

# Cache-bust marker: forces Render to rebuild from scratch so PYTHONPATH env
# (needed for `from app.models.schemas import ...`) is not served from a
# cached layer that predates the fix. Bump this comment on each redeploy.
# BUILD-CACHE-BUST-2026-10-11-FIX-PIN-4571

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/backend \
    SIDEQUEST_WEB_DIR=/app/web \
    SIDEQUEST_DATA_DIR=/app/data

WORKDIR /app

# curl is used by the container health check.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY backend ./backend
COPY web ./web
COPY eval ./eval
COPY scripts ./scripts

RUN pip install --no-deps -e . \
 && pip install fastapi "uvicorn[standard]" pydantic pillow

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
