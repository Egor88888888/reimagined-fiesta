"""
DocLens ML Database Models
Tables for storing corrections, training data, and model metadata.
"""
import uuid
import enum
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON,
    ForeignKey, Enum as SQLEnum, Index, BigInteger
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.models.database import Base


# ============================================================
# ENUMS
# ============================================================

class CorrectionStatus(str, enum.Enum):
    PENDING = "pending"          # Awaiting review
    APPROVED = "approved"        # Verified and ready for training
    REJECTED = "rejected"        # False correction
    USED_IN_TRAINING = "trained" # Already used in a training run


class TrainingStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelStatus(str, enum.Enum):
    TRAINING = "training"
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"


# ============================================================
# FIELD CORRECTION (User feedback)
# ============================================================

class FieldCorrection(Base):
    """Stores user corrections of OCR-extracted fields.

    Each row represents a single field correction:
    the original OCR output vs. what the user says is correct.
    """
    __tablename__ = "field_corrections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    recognition_id = Column(UUID(as_uuid=True), ForeignKey("recognitions.id"), nullable=False)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)

    # What was corrected
    document_type = Column(String(50), nullable=False)      # e.g. "passport_rf"
    field_name = Column(String(100), nullable=False)         # e.g. "last_name"
    original_value = Column(Text, nullable=False)            # OCR output
    corrected_value = Column(Text, nullable=False)           # User's correction
    original_confidence = Column(Float, default=0.0)         # OCR confidence

    # Context for learning
    ocr_raw_text = Column(Text)                              # Raw OCR text around this field
    image_region_hash = Column(String(64))                   # Hash of image region (for dedup)

    # Status
    status = Column(
        SQLEnum(CorrectionStatus),
        default=CorrectionStatus.PENDING,
        nullable=False
    )
    reviewed_at = Column(DateTime(timezone=True))
    reviewed_by = Column(String(255))                        # Admin who reviewed

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_corrections_doc_field", "document_type", "field_name"),
        Index("ix_corrections_status", "status"),
        Index("ix_corrections_tenant", "tenant_id"),
        Index("ix_corrections_recognition", "recognition_id"),
    )


# ============================================================
# CORRECTION PATTERN (Learned error patterns)
# ============================================================

class CorrectionPattern(Base):
    """Aggregated error patterns learned from corrections.

    Example: OCR often reads "Б" as "6" in passport last names.
    Patterns are auto-generated from FieldCorrection data.
    """
    __tablename__ = "correction_patterns"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    document_type = Column(String(50), nullable=False)
    field_name = Column(String(100), nullable=False)

    # Pattern details
    error_pattern = Column(String(500), nullable=False)     # What OCR produces wrong
    correction = Column(String(500), nullable=False)        # What it should be
    pattern_type = Column(String(50), nullable=False)       # "char_substitution", "format", "regex"

    # Statistics
    occurrence_count = Column(Integer, default=1)
    confidence = Column(Float, default=0.5)                 # How reliable this pattern is
    last_seen_at = Column(DateTime(timezone=True))

    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        Index("ix_patterns_doc_field", "document_type", "field_name"),
        Index("ix_patterns_active", "is_active"),
    )


# ============================================================
# TRAINING RUN (Model training history)
# ============================================================

class TrainingRun(Base):
    """Records each training run of the ML corrector."""
    __tablename__ = "training_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    status = Column(SQLEnum(TrainingStatus), default=TrainingStatus.QUEUED, nullable=False)

    # Training data stats
    corrections_count = Column(Integer, default=0)          # How many corrections used
    patterns_generated = Column(Integer, default=0)         # How many patterns extracted

    # Model info
    model_version = Column(String(50))                      # e.g. "v1.2.0"
    model_path = Column(String(1000))                       # Path to saved model weights

    # Metrics
    accuracy_before = Column(Float)                         # Accuracy on test set before training
    accuracy_after = Column(Float)                          # Accuracy on test set after training
    field_accuracies = Column(JSON, default=dict)           # Per-field accuracy breakdown

    # Timing
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    duration_seconds = Column(Integer)

    # Error info
    error_message = Column(Text)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_training_status", "status"),
    )


# ============================================================
# ML MODEL REGISTRY
# ============================================================

class MLModel(Base):
    """Registry of trained ML models."""
    __tablename__ = "ml_models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    name = Column(String(255), nullable=False)              # e.g. "field_corrector_passport_rf"
    version = Column(String(50), nullable=False)
    status = Column(SQLEnum(ModelStatus), default=ModelStatus.TRAINING, nullable=False)

    # What this model does
    document_type = Column(String(50))                      # null = all types
    description = Column(Text)

    # Storage
    model_path = Column(String(1000))                       # S3 or local path
    model_size_bytes = Column(BigInteger)

    # Performance
    training_run_id = Column(UUID(as_uuid=True), ForeignKey("training_runs.id"))
    accuracy = Column(Float)
    f1_score = Column(Float)

    # Metadata
    config = Column(JSON, default=dict)                     # Model hyperparameters
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    activated_at = Column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_models_name_version", "name", "version", unique=True),
        Index("ix_models_status", "status"),
    )
