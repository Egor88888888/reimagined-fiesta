"""
DocLens API Routes
"""
import uuid
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Query, Request
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import (
    Tenant, ApiKey, Recognition, UsageRecord,
    RecognitionStatus, DocumentType, PlanType,
    get_db, async_session
)
from app.api.auth import get_current_tenant, hash_api_key, record_usage
from app.api.schemas import (
    TenantCreate, TenantResponse, ApiKeyCreate, ApiKeyResponse, ApiKeyWithSecret,
    RecognitionResponse, RecognitionListItem, UsageSummary, HealthResponse,
)
from app.core.orchestrator import get_pipeline
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


# ============================================================
# HEALTH
# ============================================================

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check."""
    db_status = "ok"
    redis_status = "ok"
    ocr_ready = True

    try:
        async with async_session() as session:
            await session.execute(select(func.count()).select_from(Tenant))
    except Exception:
        db_status = "error"

    try:
        from app.api.auth import get_redis
        rd = await get_redis()
        await rd.ping()
    except Exception:
        redis_status = "error"

    return HealthResponse(
        status="ok" if db_status == "ok" else "degraded",
        version=settings.APP_VERSION,
        ocr_ready=ocr_ready,
        database=db_status,
        redis=redis_status,
    )


# ============================================================
# TENANT MANAGEMENT
# ============================================================

@router.post("/tenants", response_model=ApiKeyWithSecret, tags=["Tenants"])
async def register_tenant(data: TenantCreate, db: AsyncSession = Depends(get_db)):
    """Register a new tenant and get an API key."""
    # Check if email already exists
    result = await db.execute(select(Tenant).where(Tenant.email == data.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    # Create tenant
    tenant = Tenant(name=data.name, email=data.email, plan=PlanType.FREE)
    db.add(tenant)
    await db.flush()

    # Generate API key
    raw_key = ApiKey.generate_key()
    prefix = raw_key[8:16]  # after "dl_live_"
    key_hash = hash_api_key(raw_key)

    api_key = ApiKey(
        tenant_id=tenant.id,
        key_prefix=prefix,
        key_hash=key_hash,
        name="Default",
    )
    db.add(api_key)
    await db.commit()

    return ApiKeyWithSecret(
        id=api_key.id,
        name=api_key.name,
        key_prefix=prefix,
        created_at=api_key.created_at or datetime.now(timezone.utc),
        api_key=raw_key,  # Only shown once!
    )


@router.get("/tenants/me", response_model=TenantResponse, tags=["Tenants"])
async def get_current_tenant_info(
    tenant: Tenant = Depends(get_current_tenant),
):
    """Get current tenant info."""
    return TenantResponse(
        id=tenant.id,
        name=tenant.name,
        email=tenant.email,
        plan=tenant.plan.value,
        is_active=tenant.is_active,
        created_at=tenant.created_at,
    )


# ============================================================
# API KEYS
# ============================================================

@router.post("/api-keys", response_model=ApiKeyWithSecret, tags=["API Keys"])
async def create_api_key(
    data: ApiKeyCreate,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """Create a new API key for the tenant."""
    raw_key = ApiKey.generate_key()
    prefix = raw_key[8:16]
    key_hash = hash_api_key(raw_key)

    api_key = ApiKey(
        tenant_id=tenant.id,
        key_prefix=prefix,
        key_hash=key_hash,
        name=data.name,
    )
    db.add(api_key)
    await db.commit()

    return ApiKeyWithSecret(
        id=api_key.id,
        name=api_key.name,
        key_prefix=prefix,
        created_at=api_key.created_at or datetime.now(timezone.utc),
        api_key=raw_key,
    )


@router.get("/api-keys", response_model=list[ApiKeyResponse], tags=["API Keys"])
async def list_api_keys(
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """List all API keys for the tenant."""
    result = await db.execute(
        select(ApiKey).where(ApiKey.tenant_id == tenant.id).order_by(desc(ApiKey.created_at))
    )
    return result.scalars().all()


@router.delete("/api-keys/{key_id}", tags=["API Keys"])
async def revoke_api_key(
    key_id: uuid.UUID,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """Revoke an API key."""
    result = await db.execute(
        select(ApiKey).where(ApiKey.id == key_id, ApiKey.tenant_id == tenant.id)
    )
    api_key = result.scalar_one_or_none()
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")

    api_key.is_active = False
    await db.commit()
    return {"status": "revoked"}


# ============================================================
# DOCUMENT RECOGNITION
# ============================================================

@router.post("/recognize", response_model=RecognitionResponse, tags=["Recognition"])
async def recognize_document(
    request: Request,
    file: UploadFile = File(...),
    document_type: str = Form(default=None),
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """Recognize a document and extract structured data.

    Upload an image (JPEG, PNG) or PDF of a document.
    Optionally specify `document_type` hint:
    - passport_rf
    - passport_cis
    - driver_license
    - snils
    - inn
    """
    # Validate file
    if not file.content_type:
        raise HTTPException(status_code=400, detail="File content type is required")

    allowed_types = ["image/jpeg", "image/png", "image/tiff", "application/pdf"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {', '.join(allowed_types)}"
        )

    image_bytes = await file.read()
    if len(image_bytes) > settings.OCR_MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large. Max: {settings.OCR_MAX_FILE_SIZE // 1024 // 1024}MB")

    # Create recognition record
    recognition = Recognition(
        tenant_id=tenant.id,
        status=RecognitionStatus.PROCESSING,
        document_type_hint=document_type,
        original_filename=file.filename,
        file_size=len(image_bytes),
        mime_type=file.content_type,
        ip_address=request.client.host if request.client else None,
    )
    db.add(recognition)
    await db.flush()

    # Process document
    try:
        pipeline = get_pipeline()
        result = pipeline.process(image_bytes, document_type_hint=document_type)

        # Update recognition record
        recognition.status = RecognitionStatus.COMPLETED
        recognition.document_type = DocumentType(result.document_type) if result.document_type != "unknown" else DocumentType.UNKNOWN
        recognition.fields = result.fields
        recognition.overall_confidence = result.overall_confidence
        recognition.validation_results = result.validation
        recognition.warnings = result.warnings
        recognition.processing_time_ms = result.processing_time_ms
        recognition.completed_at = datetime.now(timezone.utc)

        await db.commit()

        # Record usage
        await record_usage(tenant.id, result.processing_time_ms, True, db)

        return RecognitionResponse(
            id=recognition.id,
            status=recognition.status.value,
            document_type=result.document_type,
            overall_confidence=result.overall_confidence,
            fields=result.fields,
            validation=result.validation,
            warnings=result.warnings,
            processing_time_ms=result.processing_time_ms,
            created_at=recognition.created_at,
            completed_at=recognition.completed_at,
        )

    except Exception as e:
        logger.exception(f"Recognition failed: {e}")
        recognition.status = RecognitionStatus.FAILED
        recognition.warnings = [str(e)]
        await db.commit()

        await record_usage(tenant.id, 0, False, db)

        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")


@router.get("/recognitions", response_model=list[RecognitionListItem], tags=["Recognition"])
async def list_recognitions(
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    status: str = Query(default=None),
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """List recognition history."""
    query = select(Recognition).where(
        Recognition.tenant_id == tenant.id
    ).order_by(desc(Recognition.created_at)).limit(limit).offset(offset)

    if status:
        query = query.where(Recognition.status == RecognitionStatus(status))

    result = await db.execute(query)
    return result.scalars().all()


@router.get("/recognitions/{recognition_id}", response_model=RecognitionResponse, tags=["Recognition"])
async def get_recognition(
    recognition_id: uuid.UUID,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific recognition result."""
    result = await db.execute(
        select(Recognition).where(
            Recognition.id == recognition_id,
            Recognition.tenant_id == tenant.id,
        )
    )
    recognition = result.scalar_one_or_none()
    if not recognition:
        raise HTTPException(status_code=404, detail="Recognition not found")
    return recognition


# ============================================================
# USAGE / BILLING
# ============================================================

@router.get("/usage", response_model=UsageSummary, tags=["Usage"])
async def get_usage(
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """Get usage summary for current billing period."""
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    period_start = today.replace(day=1)  # Start of month

    # Today's usage from Redis
    today_used = 0
    try:
        from app.api.auth import get_redis
        rd = await get_redis()
        today_key = f"ratelimit:{tenant.id}:{today.strftime('%Y-%m-%d')}"
        val = await rd.get(today_key)
        today_used = int(val) if val else 0
    except Exception:
        pass

    # Period totals from DB
    result = await db.execute(
        select(
            func.sum(UsageRecord.request_count),
            func.sum(UsageRecord.successful_count),
            func.sum(UsageRecord.failed_count),
            func.sum(UsageRecord.total_processing_ms),
        ).where(
            UsageRecord.tenant_id == tenant.id,
            UsageRecord.date >= period_start,
        )
    )
    row = result.one()
    total = row[0] or 0
    successful = row[1] or 0
    failed = row[2] or 0
    total_ms = row[3] or 0

    avg_ms = total_ms / total if total > 0 else 0

    return UsageSummary(
        plan=tenant.plan.value,
        daily_limit=tenant.daily_limit,
        today_used=today_used,
        today_remaining=max(0, tenant.daily_limit - today_used),
        period_total=total,
        period_successful=successful,
        period_failed=failed,
        avg_processing_ms=round(avg_ms, 1),
    )
