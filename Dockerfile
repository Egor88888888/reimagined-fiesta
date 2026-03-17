# ============================================================
# DocLens — Multi-stage Production Dockerfile
# Optimized for Russian document recognition (OCR)
# ============================================================

# ============================================================
# Stage 1: Builder
# ============================================================
FROM python:3.11-slim as builder

# Install system dependencies for OCR & image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-rus \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libmagic1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Create wheels for all dependencies (faster layer caching)
RUN pip install --no-cache-dir --user --no-warn-script-location \
    --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --user --no-warn-script-location \
    --wheel-dir /app/wheels -r requirements.txt

# ============================================================
# Stage 2: Runtime
# ============================================================
FROM python:3.11-slim

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-rus \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Download optimal Russian Tesseract language data (tessdata_best)
RUN mkdir -p /usr/share/tessdata && \
    curl -sSL https://github.com/tesseract-ocr/tessdata_best/raw/main/rus.traineddata \
    -o /usr/share/tessdata/rus.traineddata && \
    chmod 644 /usr/share/tessdata/rus.traineddata

# Create non-root user
RUN useradd -m -u 1000 doclens

WORKDIR /app

# Copy wheels from builder
COPY --from=builder /app/wheels /wheels

# Install Python dependencies from wheels
RUN pip install --no-cache-dir --user --no-warn-script-location \
    --no-index --find-links=/wheels /wheels/* && \
    rm -rf /wheels

# Copy application code
COPY --chown=doclens:doclens . .

# Create necessary directories
RUN mkdir -p static/css static/js templates/demo && \
    chown -R doclens:doclens /app

# Switch to non-root user
USER doclens

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default port
EXPOSE 8000

# Default command (single worker for local, override with -w for production)
CMD ["uvicorn", "run_local:app", "--host", "0.0.0.0", "--port", "8000"]
