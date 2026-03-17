"""
DocLens API Schemas (Pydantic models)
"""
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from uuid import UUID
from typing import Optional


# ============================================================
# Tenant / Auth
# ============================================================

class TenantCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=255)
    email: str = Field(..., min_length=5, max_length=255)


class TenantResponse(BaseModel):
    id: UUID
    name: str
    email: str
    plan: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class ApiKeyCreate(BaseModel):
    name: str = Field(default="Default", max_length=255)


class ApiKeyResponse(BaseModel):
    id: UUID
    name: str
    key_prefix: str
    created_at: datetime

    class Config:
        from_attributes = True


class ApiKeyWithSecret(ApiKeyResponse):
    """Returned only on creation — contains the full key."""
    api_key: str  # Only shown once!


# ============================================================
# Recognition
# ============================================================

class RecognitionResponse(BaseModel):
    id: UUID
    status: str
    document_type: Optional[str] = None
    overall_confidence: float = 0.0
    fields: dict = {}
    validation: dict = {}
    warnings: list[str] = []
    processing_time_ms: Optional[int] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class RecognitionListItem(BaseModel):
    id: UUID
    status: str
    document_type: Optional[str] = None
    overall_confidence: float = 0.0
    original_filename: Optional[str] = None
    processing_time_ms: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ============================================================
# Usage / Billing
# ============================================================

class UsageResponse(BaseModel):
    date: str
    request_count: int
    successful_count: int
    failed_count: int
    avg_processing_ms: float


class UsageSummary(BaseModel):
    plan: str
    daily_limit: int
    today_used: int
    today_remaining: int
    period_total: int
    period_successful: int
    period_failed: int
    avg_processing_ms: float


# ============================================================
# Health
# ============================================================

class HealthResponse(BaseModel):
    status: str
    version: str
    ocr_ready: bool
    database: str
    redis: str
