FROM python:3.11-slim

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

RUN mkdir -p /usr/share/tessdata && \
    curl -sSL https://github.com/tesseract-ocr/tessdata_best/raw/main/rus.traineddata \
    -o /usr/share/tessdata/rus.traineddata && \
    chmod 644 /usr/share/tessdata/rus.traineddata

RUN useradd -m -u 1000 doclens

WORKDIR /app

COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

COPY --chown=doclens:doclens . .

RUN mkdir -p static templates && \
    chown -R doclens:doclens /app

USER doclens

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/v1/health || exit 1

EXPOSE 8000

CMD uvicorn run_local:app --host 0.0.0.0 --port ${PORT:-8000}
