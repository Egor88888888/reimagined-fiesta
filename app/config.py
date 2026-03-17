"""
DocLens Configuration
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    APP_NAME: str = "DocLens"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    SECRET_KEY: str = "change-me-in-production-use-openssl-rand-hex-32"
    ALLOWED_ORIGINS: list[str] = ["*"]

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://doclens:doclens@localhost:5432/doclens"
    DATABASE_URL_SYNC: str = "postgresql://doclens:doclens@localhost:5432/doclens"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # Storage (S3/MinIO)
    S3_ENDPOINT: str = "http://localhost:9000"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET: str = "doclens-documents"
    S3_REGION: str = "us-east-1"

    # OCR Settings
    OCR_USE_GPU: bool = False
    OCR_DET_DB_THRESH: float = 0.3
    OCR_REC_BATCH_NUM: int = 6
    OCR_MAX_IMAGE_SIZE: int = 4096  # max dimension in pixels
    OCR_SUPPORTED_FORMATS: list[str] = ["image/jpeg", "image/png", "image/tiff", "application/pdf"]
    OCR_MAX_FILE_SIZE: int = 20 * 1024 * 1024  # 20MB

    # Confidence thresholds
    CONFIDENCE_AUTO_FILL: float = 0.95
    CONFIDENCE_REVIEW: float = 0.70

    # Rate limiting (per tenant)
    RATE_LIMIT_FREE: int = 100        # requests per day
    RATE_LIMIT_BASIC: int = 1000
    RATE_LIMIT_PRO: int = 10000
    RATE_LIMIT_ENTERPRISE: int = 100000

    # JWT
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24

    # ML / Active Learning
    ML_ENABLED: bool = True                        # Enable ML post-processing
    ML_MODEL_DIR: str = "/app/ml_models"          # Directory for model artifacts
    ML_MIN_CORRECTIONS_FOR_TRAINING: int = 20     # Min corrections to start training
    ML_AUTO_TRAIN_THRESHOLD: int = 100            # Auto-trigger training at N corrections
    ML_PATTERN_CONFIDENCE_THRESHOLD: float = 0.6  # Min confidence to apply a pattern
    ML_AUTO_APPROVE_COUNT: int = 3                # Auto-approve after N identical corrections

    # External APIs (optional enrichment)
    FMS_CHECK_ENABLED: bool = True
    FNS_CHECK_ENABLED: bool = True
    DADATA_API_KEY: str = ""
    DADATA_SECRET: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
