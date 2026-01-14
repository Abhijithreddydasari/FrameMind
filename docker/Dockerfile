# FrameMind API Server
# Multi-stage build for optimized production image

FROM python:3.11-slim as base

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============ Builder Stage ============
FROM base as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install pip and build tools
RUN pip install --upgrade pip setuptools wheel

# Copy dependency file
COPY pyproject.toml ./

# Install dependencies (without dev dependencies)
RUN pip install --no-cache-dir . \
    && pip install --no-cache-dir .[postgres]

# ============ Production Stage ============
FROM base as production

# Create non-root user
RUN groupadd -r framemind && useradd -r -g framemind framemind

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/

# Create data directory
RUN mkdir -p /app/data && chown -R framemind:framemind /app

# Switch to non-root user
USER framemind

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ============ Development Stage ============
FROM base as development

# Install dev dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]"

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run with reload for development
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
