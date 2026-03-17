"""
DocLens ML API Routes
Endpoints for feedback, corrections, training, and ML system management.
"""
import logging
from datetime import datetime, timezone
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import Tenant, Recognition, get_db, async_session
from app.api.auth import get_current_tenant
from app.ml.models import (
    FieldCorrection, CorrectionPattern, TrainingRun,
    CorrectionStatus, TrainingStatus,
)
from app.ml.feedback import get_feedback_collector
from app.ml.trainer import get_trainer
from app.ml.corrector import get_corrector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml", tags=["ML & Learning"])


# ============================================================
# SCHEMAS
# ============================================================

class FieldCorrectionSubmit(BaseModel):
    """Single field correction."""
    field_name: str = Field(..., min_length=1)
    original_value: str
    corrected_value: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class BulkCorrectionSubmit(BaseModel):
    """Submit corrections for multiple fields of a recognition."""
    recognition_id: str
    document_type: str
    corrections: list[FieldCorrectionSubmit]


class CorrectionResponse(BaseModel):
    id: UUID
    field_name: str
    original_value: str
    corrected_value: str
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class TrainingRequest(BaseModel):
    document_type: str | None = None
    force: bool = False


class TrainingRunResponse(BaseModel):
    id: UUID
    status: str
    corrections_count: int
    patterns_generated: int
    accuracy_before: float | None
    accuracy_after: float | None
    model_version: str | None
    duration_seconds: int | None
    error_message: str | None
    created_at: datetime

    class Config:
        from_attributes = True


class PatternResponse(BaseModel):
    id: UUID
    document_type: str
    field_name: str
    error_pattern: str
    correction: str
    pattern_type: str
    occurrence_count: int
    confidence: float
    is_active: bool

    class Config:
        from_attributes = True


# ============================================================
# FEEDBACK / CORRECTIONS
# ============================================================

@router.post("/corrections", tags=["ML & Learning"])
async def submit_corrections(
    data: BulkCorrectionSubmit,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """Submit field corrections for a recognition result.

    This is the primary learning input. When a user corrects
    a field, the system learns from the correction to improve
    future results.

    The correction is auto-approved if the same correction
    has been seen 3+ times before.
    """
    # Verify recognition belongs to tenant
    result = await db.execute(
        select(Recognition).where(
            Recognition.id == data.recognition_id,
            Recognition.tenant_id == tenant.id,
        )
    )
    recognition = result.scalar_one_or_none()
    if not recognition:
        raise HTTPException(status_code=404, detail="Recognition not found")

    feedback = get_feedback_collector()
    corrections_dict = {}
    for c in data.corrections:
        corrections_dict[c.field_name] = {
            "original": c.original_value,
            "corrected": c.corrected_value,
            "confidence": c.confidence,
        }

    saved = await feedback.submit_bulk_corrections(
        db=db,
        recognition_id=data.recognition_id,
        tenant_id=str(tenant.id),
        document_type=data.document_type,
        corrections=corrections_dict,
    )

    await db.commit()

    return {
        "status": "ok",
        "corrections_saved": len(saved),
        "message": f"Saved {len(saved)} corrections for learning",
    }


@router.get("/corrections", response_model=list[CorrectionResponse])
async def list_corrections(
    document_type: str = Query(default=None),
    field_name: str = Query(default=None),
    status: str = Query(default=None),
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """List submitted corrections with filtering."""
    query = select(FieldCorrection).where(
        FieldCorrection.tenant_id == tenant.id
    ).order_by(FieldCorrection.created_at.desc()).limit(limit).offset(offset)

    if document_type:
        query = query.where(FieldCorrection.document_type == document_type)
    if field_name:
        query = query.where(FieldCorrection.field_name == field_name)
    if status:
        query = query.where(FieldCorrection.status == CorrectionStatus(status))

    result = await db.execute(query)
    return result.scalars().all()


@router.get("/corrections/stats")
async def get_correction_stats(
    document_type: str = Query(default=None),
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """Get correction statistics: how many corrections per field, per type."""
    feedback = get_feedback_collector()
    return await feedback.get_correction_stats(db, document_type)


@router.put("/corrections/{correction_id}/approve")
async def approve_correction(
    correction_id: UUID,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """Manually approve a correction for training."""
    result = await db.execute(
        select(FieldCorrection).where(
            FieldCorrection.id == correction_id,
            FieldCorrection.tenant_id == tenant.id,
        )
    )
    correction = result.scalar_one_or_none()
    if not correction:
        raise HTTPException(status_code=404, detail="Correction not found")

    correction.status = CorrectionStatus.APPROVED
    correction.reviewed_at = datetime.now(timezone.utc)
    await db.commit()

    return {"status": "approved", "id": str(correction_id)}


@router.put("/corrections/{correction_id}/reject")
async def reject_correction(
    correction_id: UUID,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """Reject a correction (mark as false)."""
    result = await db.execute(
        select(FieldCorrection).where(
            FieldCorrection.id == correction_id,
            FieldCorrection.tenant_id == tenant.id,
        )
    )
    correction = result.scalar_one_or_none()
    if not correction:
        raise HTTPException(status_code=404, detail="Correction not found")

    correction.status = CorrectionStatus.REJECTED
    correction.reviewed_at = datetime.now(timezone.utc)
    await db.commit()

    return {"status": "rejected", "id": str(correction_id)}


# ============================================================
# PATTERNS
# ============================================================

@router.get("/patterns", response_model=list[PatternResponse])
async def list_patterns(
    document_type: str = Query(default=None),
    field_name: str = Query(default=None),
    limit: int = Query(default=100, le=500),
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """List learned correction patterns."""
    query = select(CorrectionPattern).where(
        CorrectionPattern.is_active == True
    ).order_by(CorrectionPattern.occurrence_count.desc()).limit(limit)

    if document_type:
        query = query.where(CorrectionPattern.document_type == document_type)
    if field_name:
        query = query.where(CorrectionPattern.field_name == field_name)

    result = await db.execute(query)
    return result.scalars().all()


@router.post("/patterns/extract")
async def extract_patterns(
    document_type: str = Query(default=None),
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """Manually trigger pattern extraction from corrections."""
    feedback = get_feedback_collector()
    patterns = await feedback.extract_patterns(db, document_type)
    await db.commit()

    return {
        "status": "ok",
        "patterns_extracted": len(patterns),
    }


@router.put("/patterns/{pattern_id}/toggle")
async def toggle_pattern(
    pattern_id: UUID,
    active: bool = Query(...),
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """Enable or disable a correction pattern."""
    result = await db.execute(
        select(CorrectionPattern).where(CorrectionPattern.id == pattern_id)
    )
    pattern = result.scalar_one_or_none()
    if not pattern:
        raise HTTPException(status_code=404, detail="Pattern not found")

    pattern.is_active = active
    await db.commit()

    # Reload corrector patterns cache
    corrector = get_corrector()
    await corrector.load_patterns(db)

    return {"status": "ok", "pattern_id": str(pattern_id), "active": active}


# ============================================================
# TRAINING
# ============================================================

@router.post("/train", response_model=TrainingRunResponse)
async def start_training(
    data: TrainingRequest,
    background_tasks: BackgroundTasks,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """Start a model training run.

    Trains the ML corrector on accumulated corrections.
    Requires at least 20 corrections (or use force=true).

    The training:
    1. Extracts error patterns from corrections
    2. Builds a character substitution probability matrix
    3. Creates per-field error models
    4. Evaluates accuracy improvement
    5. Saves the new model

    After training, the corrector automatically loads the new model.
    """
    trainer = get_trainer()
    training_run = await trainer.run_training(
        db=db,
        document_type=data.document_type,
        force=data.force,
    )

    # Reload corrector with new model
    if training_run.status == TrainingStatus.COMPLETED:
        corrector = get_corrector()
        corrector.load_ml_model()
        await corrector.load_patterns(db)

    return training_run


@router.get("/train/status")
async def get_training_status(
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """Get ML system status: last training, active models, pending data."""
    trainer = get_trainer()
    return await trainer.get_training_status(db)


@router.get("/train/history", response_model=list[TrainingRunResponse])
async def list_training_runs(
    limit: int = Query(default=20, le=100),
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """List training run history."""
    result = await db.execute(
        select(TrainingRun)
        .order_by(TrainingRun.created_at.desc())
        .limit(limit)
    )
    return result.scalars().all()


# ============================================================
# CORRECTOR (Real-time)
# ============================================================

@router.post("/correct")
async def correct_fields_endpoint(
    document_type: str,
    fields: dict,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """Apply ML corrections to fields (for testing/debugging).

    Send extracted fields and get back corrected versions.
    Useful for testing how the corrector would modify OCR output.
    """
    corrector = get_corrector()
    corrected = await corrector.correct_fields(db, document_type, fields)
    return {
        "document_type": document_type,
        "original_fields": fields,
        "corrected_fields": corrected,
    }


@router.post("/reload")
async def reload_ml_models(
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """Force reload ML model and patterns cache."""
    corrector = get_corrector()
    corrector.load_ml_model()
    await corrector.load_patterns(db)

    return {
        "status": "ok",
        "message": "ML model and patterns reloaded",
        "ml_model_loaded": corrector._ml_model_loaded,
        "patterns_loaded": corrector._patterns_loaded,
    }
