---
title: DocLens
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# DocLens — Russian Document Recognition SaaS

A production-ready FastAPI-based SaaS platform for automated recognition and extraction of Russian official documents using OCR (Tesseract + EasyOCR). Supports passports, driving licenses, SNILS, and INN documents.

## Features

- **Multi-document support**: Russian passport (RF/CIS), driving license, SNILS, INN
- **Advanced OCR**: Tesseract with Russian language data + EasyOCR fallback
- **Document classification**: Automatic document type detection
- **Async processing**: Celery workers for background tasks
- **Production-ready**: Docker, PostgreSQL, Redis, S3/MinIO, health checks
- **API-first**: RESTful API with OpenAPI documentation
- **Demo interface**: Web UI for testing document recognition
- **Usage tracking**: Built-in metrics and usage analytics
- **Security**: API key authentication, CORS, rate limiting

## Quick Start

### Development Mode (Single Command)

```bash
bash start.sh
```

This will:
1. Check Python 3.11+
2. Install Tesseract OCR + Russian language data
3. Create Python virtual environment
4. Install dependencies
5. Start FastAPI server on `http://localhost:8000`

**Access points:**
- Demo UI: http://localhost:8000/demo
- API Docs: http://localhost:8000/docs
- Health: http://localhost:8000/api/v1/health

### Production Mode (Docker)

```bash
# Build and start full stack
docker-compose up -d

# Monitor logs
docker-compose logs -f doclens-api

# Stop services
docker-compose down
```

**Services:**
- API Server: `http://localhost:8000`
- MinIO Console: `http://localhost:9001` (minioadmin / minioadmin)
- PostgreSQL: `localhost:5432`
- Redis: `localhost:6379`

### Production Mode (Local with Gunicorn)

```bash
PRODUCTION=1 WORKERS=4 bash start.sh
```

This starts the app with 4 Gunicorn workers for production workloads.

## Configuration

### Environment Variables

Create `.env` file from template:

```bash
cp .env.example .env
```

**Key variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `LOG_LEVEL` | `info` | Logging level (debug/info/warning/error) |
| `DEMO_MODE` | `true` | Enable demo/test UI |
| `MAX_FILE_SIZE_MB` | `20` | Max upload size in MB |
| `SECRET_KEY` | (required) | 32-char hex for JWT signing |
| `DATABASE_URL` | (optional) | PostgreSQL async connection string |
| `REDIS_URL` | (optional) | Redis connection string |
| `OCR_USE_GPU` | `false` | Enable GPU acceleration (requires CUDA) |
| `CONFIDENCE_AUTO_FILL` | `0.95` | Threshold for automatic data filling |
| `CONFIDENCE_REVIEW` | `0.70` | Threshold for review flag |

## Docker Deployment

### Build Custom Image

```bash
docker build -t doclens:latest .
```

### Run Single Container (Local Dev)

```bash
docker run -it -p 8000:8000 \
  -e SECRET_KEY=your-secret-here \
  -e DEMO_MODE=true \
  doclens:latest
```

### Run Full Stack (Production)

```bash
# Set production secret
export SECRET_KEY=$(openssl rand -hex 32)

# Start all services
docker-compose up -d

# View status
docker-compose ps

# Check logs
docker-compose logs doclens-api -f

# Scale workers
docker-compose up -d --scale doclens-worker=3
```

### Container Resource Limits

Current docker-compose.yml defines:

| Service | CPU | Memory |
|---------|-----|--------|
| API | 2 | 2G |
| Worker | 1 | 1G |
| Database | - | 1G |
| Redis | - | 512M |
| MinIO | - | 512M |

Adjust in `docker-compose.yml` under `deploy.resources`.

## API Documentation

### Base URL

```
http://localhost:8000/api/v1
```

### Endpoints

#### Health Check

```bash
GET /api/v1/health
```

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2026-03-16T12:00:00Z",
  "version": "1.0.0"
}
```

#### Recognize Document

```bash
POST /api/v1/recognize
Content-Type: multipart/form-data

file: <image-file>
document_type: "auto" (optional, auto|passport_rf|passport_cis|driver_license|snils|inn)
```

**cURL Example:**

```bash
curl -X POST "http://localhost:8000/api/v1/recognize" \
  -H "X-API-Key: your-api-key" \
  -F "file=@passport.png" \
  -F "document_type=auto"
```

**Response (Passport):**
```json
{
  "status": "success",
  "document_type": "passport_rf",
  "confidence": 0.96,
  "data": {
    "surname": "ИВАНОВ",
    "given_name": "ИВАН",
    "patronymic": "ИВАНОВИЧ",
    "gender": "М",
    "date_of_birth": "01.01.1980",
    "place_of_birth": "г. Москва",
    "document_number": "1234567890",
    "issued_date": "10.05.2015",
    "valid_until": "10.05.2025"
  },
  "ocr_text": "...",
  "processing_time_ms": 1234
}
```

#### Get Usage Statistics

```bash
GET /api/v1/usage
Authorization: Bearer <jwt-token>
```

**Response:**
```json
{
  "period": "2026-03-01T00:00:00Z/2026-03-16T23:59:59Z",
  "total_requests": 1234,
  "successful_recognitions": 1100,
  "failed_recognitions": 134,
  "documents_by_type": {
    "passport_rf": 600,
    "driver_license": 300,
    "snils": 150,
    "inn": 50
  },
  "average_processing_time_ms": 2500
}
```

### Authentication

#### API Key (Header)

```bash
curl -H "X-API-Key: dl_live_xxxxx" \
  http://localhost:8000/api/v1/health
```

#### Bearer Token (Authorization Header)

```bash
curl -H "Authorization: Bearer eyJhbGciOiJ..." \
  http://localhost:8000/api/v1/health
```

### Error Handling

**400 Bad Request:**
```json
{
  "detail": "Invalid file format. Accepted: PNG, JPG, PDF"
}
```

**401 Unauthorized:**
```json
{
  "detail": "Invalid or missing API key"
}
```

**413 Payload Too Large:**
```json
{
  "detail": "File size exceeds 20MB limit"
}
```

**429 Too Many Requests:**
```json
{
  "detail": "Rate limit exceeded. Retry after 60 seconds"
}
```

**503 Service Unavailable:**
```json
{
  "detail": "OCR engine not available. Try again later"
}
```

## Testing

### Run API Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run all tests
pytest tests/test_api.py -v

# Run specific test
pytest tests/test_api.py::test_health_endpoint -v

# With coverage
pip install pytest-cov
pytest tests/test_api.py --cov=. --cov-report=html
```

### Test Coverage

The test suite includes:

- **Health checks**: Endpoint availability and response format
- **Demo UI**: Page loads and form functionality
- **Document recognition**: File upload, image validation, OCR processing
- **Authentication**: API key validation, Bearer tokens
- **Usage tracking**: Metrics and statistics endpoints
- **Error handling**: 4xx/5xx responses, rate limiting
- **Performance**: Response time assertions

### Integration Testing

```bash
# Start full stack
docker-compose up -d

# Run integration tests (requires running services)
pytest tests/test_api.py -v -m integration
```

## Architecture Overview

### Component Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Server                       │
│  - Request handling, routing, validation               │
│  - API endpoints, health checks, demo UI               │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┼──────────┬─────────────┐
        │          │          │             │
        ▼          ▼          ▼             ▼
    OCR Engine  Database   Cache        Document
    (Tesseract) (PostgreSQL) (Redis)     Storage
    (EasyOCR)                            (MinIO)
        │
    ┌───┴──────────────────┐
    │                      │
    ▼                      ▼
  Classifier          Parsers
  (Doc Type)      (Field Extraction)
```

### Data Flow

```
User Upload
    ↓
Validation (size, format, MIME type)
    ↓
Document Classification (type detection)
    ↓
Preprocessing (rotation, enhancement, upscaling)
    ↓
OCR Engine (Tesseract or EasyOCR)
    ↓
Field Extraction (parser for specific doc type)
    ↓
Validation (MRZ, checksums, confidence)
    ↓
Response to user
    ↓
Storage (PostgreSQL + MinIO)
```

### Supported Document Types

| Type | Code | MRZ | Fields |
|------|------|-----|--------|
| Russian Passport | `passport_rf` | ✓ | Surname, Name, Patronymic, DoB, Gender, Serial, Issue/Expiry |
| CIS Passport | `passport_cis` | ✓ | Same as RF + Country |
| Driving License | `driver_license` | ✗ | Categories, Issue/Expiry, Address |
| SNILS (Insurance) | `snils` | ✗ | Insurance number, Name, DoB |
| INN (Tax ID) | `inn` | ✗ | 10-digit tax ID |

## Database Schema

### Key Tables

```sql
-- Tenants
CREATE TABLE tenants (
  id UUID PRIMARY KEY,
  name VARCHAR(255),
  email VARCHAR(255) UNIQUE,
  plan VARCHAR(50),  -- free, pro, enterprise
  is_active BOOLEAN,
  created_at TIMESTAMP
);

-- API Keys
CREATE TABLE api_keys (
  id UUID PRIMARY KEY,
  tenant_id UUID REFERENCES tenants(id),
  key_hash VARCHAR(255) UNIQUE,
  name VARCHAR(255),
  is_active BOOLEAN,
  created_at TIMESTAMP
);

-- Recognition Results
CREATE TABLE recognitions (
  id UUID PRIMARY KEY,
  tenant_id UUID REFERENCES tenants(id),
  document_type VARCHAR(50),
  confidence DECIMAL(3,2),
  raw_data JSONB,
  parsed_data JSONB,
  processing_time_ms INTEGER,
  created_at TIMESTAMP
);

-- Usage Tracking
CREATE TABLE usage_logs (
  id UUID PRIMARY KEY,
  tenant_id UUID REFERENCES tenants(id),
  endpoint VARCHAR(255),
  document_type VARCHAR(50),
  success BOOLEAN,
  processing_time_ms INTEGER,
  created_at TIMESTAMP
);
```

## Monitoring & Logging

### Logs

```bash
# Development (console)
tail -f /tmp/doclens.log

# Docker
docker-compose logs -f doclens-api

# Structured logging (JSON format in production)
# Each log line includes: timestamp, level, service, message, context
```

### Metrics (Prometheus)

```
GET /metrics
```

Exposed metrics:
- `doclens_requests_total` — Total API requests
- `doclens_recognition_duration_seconds` — OCR processing time
- `doclens_recognition_confidence_histogram` — Confidence distribution
- `doclens_errors_total` — Error count by type

### Health Checks

```bash
# API health
curl http://localhost:8000/api/v1/health

# Database connection
curl http://localhost:8000/api/v1/health/db

# Redis connectivity
curl http://localhost:8000/api/v1/health/redis

# OCR engine status
curl http://localhost:8000/api/v1/health/ocr
```

## Performance Tuning

### Tesseract Configuration

```bash
# High-quality mode (slower, more accurate)
TESSERACT_OEM=1  # OCR Engine Mode: 1 = LSTM + Legacy

# Language data quality
# Fast: ~5MB  → Basic recognition
# Standard: ~15MB → Balanced
# Best: ~50MB → Optimal (recommended for production)

# Download best model:
curl -L -o /usr/share/tessdata/rus.traineddata \
  https://github.com/tesseract-ocr/tessdata_best/raw/main/rus.traineddata
```

### Image Preprocessing

The app automatically:
1. Detects document orientation (rotation correction)
2. Enhances contrast and brightness
3. Removes noise
4. Upscales low-resolution images

To disable preprocessing:
```bash
export SKIP_PREPROCESSING=false
```

### Memory Optimization

```bash
# Limit image processing memory
export OCR_MAX_IMAGE_SIZE=2048

# Reduce worker concurrency
export CELERY_WORKER_CONCURRENCY=1
```

### Database Connection Pooling

```python
# In production, adjust pool settings:
SQLALCHEMY_POOL_SIZE=20
SQLALCHEMY_POOL_RECYCLE=3600
SQLALCHEMY_POOL_PRE_PING=true
```

## Deployment Checklist

- [ ] Set `SECRET_KEY` to strong random value (32+ hex chars)
- [ ] Set `DEBUG=false`
- [ ] Configure `DATABASE_URL` for PostgreSQL
- [ ] Configure `REDIS_URL` for Redis
- [ ] Configure `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`
- [ ] Set appropriate resource limits
- [ ] Enable HTTPS/TLS (use reverse proxy like nginx)
- [ ] Configure CORS for your frontend domain
- [ ] Enable rate limiting
- [ ] Set up monitoring and alerting
- [ ] Configure log aggregation (ELK, CloudWatch, etc.)
- [ ] Enable database backups
- [ ] Test disaster recovery
- [ ] Document API key rotation procedure

## Troubleshooting

### Tesseract not found

```bash
# macOS
brew install tesseract tesseract-lang

# Linux (Debian/Ubuntu)
sudo apt-get install tesseract-ocr tesseract-ocr-rus

# Linux (RHEL/CentOS)
sudo yum install tesseract tesseract-langpack-rus
```

### Russian language not recognized

```bash
# Check installed languages
tesseract --list-langs

# Download better model
curl -L -o /usr/share/tessdata/rus.traineddata \
  https://github.com/tesseract-ocr/tessdata_best/raw/main/rus.traineddata
```

### OCR too slow

- Use `tessdata_fast` instead of `tessdata_best`
- Reduce `OCR_MAX_IMAGE_SIZE`
- Enable GPU: `OCR_USE_GPU=true` (requires CUDA)
- Run multiple Celery workers

### Memory issues

```bash
# Reduce model cache
export PADDLE_LITE_THREADS_NUM=2

# Limit concurrent OCR jobs
export CELERY_WORKER_CONCURRENCY=1
```

### Database connection errors

```bash
# Check PostgreSQL is running
docker-compose logs doclens-db

# Reset database
docker-compose exec doclens-db dropdb doclens
docker-compose exec doclens-db createdb doclens
```

## Contributing

When modifying code:

1. Follow PEP 8 style guide
2. Add tests for new features
3. Update README for significant changes
4. Run full test suite before commit
5. Update requirements.txt if adding dependencies

## License

Proprietary - DocLens SaaS
All rights reserved.

## Support

For issues, feature requests, or questions:

- **Email**: support@doclens.io
- **Documentation**: https://docs.doclens.io
- **API Issues**: GitHub Issues in the repo

---

**Last Updated**: March 16, 2026
