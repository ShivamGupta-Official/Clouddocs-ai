# ==============================================================================
# CloudDocs AI — Production Container Image
# Multi-purpose image for FastAPI backend and Streamlit frontend
# ==============================================================================

FROM python:3.14-slim-bookworm AS base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

# Install essential system utilities (curl for healthchecks, build tools if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Create a secure non-root user and group
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser

# Copy dependency requirements
COPY requirements.txt .

# Install CPU-optimized PyTorch first, followed by remaining requirements
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, evaluation, and configuration
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY evaluation/ ./evaluation/
COPY pyproject.toml .

# Create persistent data directories and assign ownership to appuser
RUN mkdir -p /app/data/uploads /app/data/vector_db && \
    chown -R appuser:appgroup /app

# Switch to non-root user for security
USER appuser

# Declare volume for data persistence (uploads, ChromaDB vector store, SQLite metadata)
VOLUME ["/app/data"]

# Expose FastAPI backend (8000) and Streamlit frontend (8501) ports
EXPOSE 8000 8501

# Health check against backend /health endpoint
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: launch FastAPI server with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
