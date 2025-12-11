# ---- Stage 1: build wheels ----
FROM python:3.11-slim AS build

# Install build deps (as root)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and build wheels (include dependencies)
COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip wheel -r requirements.txt --wheel-dir=/wheels

# ---- Stage 2: runtime image ----
FROM python:3.11-slim

# Install runtime OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Create non-root user but stay root for installation
RUN useradd --create-home --shell /bin/bash appuser \
 && mkdir -p /app /wheels /tmp \
 && chown -R root:root /wheels /app /tmp

# Copy built wheels from build stage
COPY --from=build /wheels /wheels

WORKDIR /app

# Install wheels (running as root) and then delete the wheel cache
RUN python -m pip install --no-cache-dir /wheels/* \
 && rm -rf /wheels

# Copy app files and set ownership to appuser
COPY service/ /app/service/
COPY service/inference_service.py /app/inference_service.py
COPY service/artifacts/ /app/artifacts/
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser
ENV HOME=/home/appuser
ENV PATH="$HOME/.local/bin:$PATH"

# rest of your ENV / CMD...
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
EXPOSE 8080
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
