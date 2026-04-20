# ============================================================
# ToolForge Containerfile (Podman)
# ============================================================
# Why Podman over Docker?
#   1. Daemonless — no root-running Docker daemon needed
#   2. OCI-compliant — images are interchangeable with Docker
#   3. Rootless by default — better security posture
#   4. Podman Compose works identically to Docker Compose
#
# Build:  podman build -t toolforge:latest -f Containerfile .
# Run:    podman run --rm -it -v ./data:/app/data:z toolforge:latest
# ============================================================

# --- Stage 1: Base with Python ---
FROM python:3.12-slim AS base

# Set environment variables for reproducible Python behavior
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies needed by ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- Stage 2: Dependencies ---
FROM base AS deps

# Copy only dependency spec first (Docker layer caching optimization)
# If pyproject.toml hasn't changed, this layer is cached
COPY pyproject.toml .
RUN pip install -e ".[train,serve]"

# --- Stage 3: Application ---
FROM deps AS app

# Copy the rest of the application
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

# Install the package itself
RUN pip install -e .

# Create directories for mounted volumes
RUN mkdir -p data/raw data/processed data/eval artifacts/models artifacts/reports

# Default: run the eval harness
ENTRYPOINT ["python", "-m", "toolforge.cli"]
CMD ["--help"]

# Health check for serving mode
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()" || exit 1

EXPOSE 8000
