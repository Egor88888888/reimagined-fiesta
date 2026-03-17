"""
DocLens Database Models
"""
import uuid
import secrets
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON,
    ForeignKey, Enum as SQLEnum, Index, BigInteger
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker
from sqlalchemy.sql import func
import enum

from app.config import get_settings

settings = get_settings()

engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


# ============================================================
# ENUMS
# ============================================================

class PlanType(str, enum.Enum):
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class DocumentType(str, enum.Enum):
    PASSPORT_RF = "passport_rf"
    PASSPORT_CIS = "passport_cis"
    DRIVER_LICENSE = "driver_license"
    SNILS = "snils"
    INN = "inn"
    UNKNOWN = "unknown"


class RecognitionStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================
# TENANT (Organization)
# ============================================================

class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    plan = Column(SQLEnum(PlanType), default=PlanType.FREE, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    api_keys = relationship("ApiKey", back_populates="tenant", cascade="all, delete-orphan")
    recognitions = relationship("Recognition", back_populates="tenant", cascade="all, delete-orphan")
    usage_records = relationship("UsageRecord", back_populates="tenant", cascade="all, delete-orphan")

    @property
    def daily_limit(self) -> int:
        limits = {
            PlanType.FREE: 100,
            PlanType.BASIC: 1000,
            PlanType.PRO: 10000,
            PlanType.ENTERPRISE: 100000,
        }
        return limits.get(self.plan, 100)


# ============================================================
# API KEY
# ============================================================

class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    key_prefix = Column(String(8), nullable=False)       # First 8 chars for lookup
    key_hash = Column(String(128), nullable=False)        # SHA-256 hash
    name = Column(String(255), default="Default")
    is_active = Column(Boolean, default=True)
    last_used_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    tenant = relationship("Tenant", back_populates="api_keys")

    __table_args__ = (
        Index("ix_api_keys_prefix", "key_prefix"),
    )

    @staticmethod
    def generate_key() -> str:
        """Generate a new API key: dl_live_<random>"""
        return f"dl_live_{secrets.token_hex(24)}"


# ============================================================
# RECOGNITION (Document recognition job)
# ============================================================

class Recognition(Base):
    __tablename__ = "recognitions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    status = Column(SQLEnum(RecognitionStatus), default=RecognitionStatus.PENDING, nullable=False)
    document_type = Column(SQLEnum(DocumentType), nullable=True)
    document_type_hint = Column(String(50), nullable=True)  # Client hint

    # File info
    original_filename = Column(String(500))
    file_path = Column(String(1000))  # S3 path
    file_size = Column(Integer)
    mime_type = Column(String(100))

    # Results
    fields = Column(JSON, default=dict)              # Extracted fields
    overall_confidence = Column(Float, default=0.0)
    validation_results = Column(JSON, default=dict)  # Validation outcomes
    warnings = Column(JSON, default=list)
    raw_ocr = Column(JSON, default=list)             # Raw OCR output
    processing_time_ms = Column(Integer)

    # Metadata
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    callback_url = Column(String(1000))              # Webhook
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))

    # Relationships
    tenant = relationship("Tenant", back_populates="recognitions")

    __table_args__ = (
        Index("ix_recognitions_tenant_status", "tenant_id", "status"),
        Index("ix_recognitions_created", "created_at"),
    )


# ============================================================
# USAGE RECORD (For billing & rate limiting)
# ============================================================

class UsageRecord(Base):
    __tablename__ = "usage_records"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    date = Column(DateTime(timezone=True), nullable=False)  # Truncated to day
    request_count = Column(Integer, default=0)
    successful_count = Column(Integer, default=0)
    failed_count = Column(Integer, default=0)
    total_processing_ms = Column(BigInteger, default=0)

    tenant = relationship("Tenant", back_populates="usage_records")

    __table_args__ = (
        Index("ix_usage_tenant_date", "tenant_id", "date", unique=True),
    )


# ============================================================
# DB Helpers
# ============================================================

async def get_db() -> AsyncSession:
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
