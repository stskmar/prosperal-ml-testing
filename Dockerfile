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

# Install runtime OS deps (if any) and minimal utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Create non-root user and appdir
RUN useradd --create-home --shell /bin/bash appuser \
 && mkdir -p /app /wheels /tmp \
 && chown -R appuser:appuser /app /wheels /tmp

# Copy built wheels from build stage
COPY --from=build /wheels /wheels

# Switch to non-root user for pip install and runtime
USER appuser
ENV HOME=/home/appuser
ENV PATH="$HOME/.local/bin:$PATH"

# Install wheels as non-root into user site-packages
# --no-deps is not used because wheels contain deps we built earlier
RUN python -m pip install --user --no-cache-dir /wheels/* \
 && rm -rf /wheels

# Copy application code (ensure ownership via current user)
COPY --chown=appuser:appuser service/ /app/service/
COPY --chown=appuser:appuser service/inference_service.py /app/inference_service.py
# copy artifacts (if needed at runtime)
COPY --chown=appuser:appuser service/artifacts/ /app/artifacts/

WORKDIR /app

# Ensure artifacts readable
RUN chmod -R a+r /app/artifacts || true

# Cloud Run expects PORT env var  (uvicorn will use this)
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV TMPDIR=/tmp

# Expose port (informational for readers; Cloud Run uses PORT env)
EXPOSE 8080

# Use a simple, production-ready command (adjust path/module as needed)
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
