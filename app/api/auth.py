"""
DocLens API Authentication & Rate Limiting
"""
import hashlib
import time
from datetime import datetime, timezone, timedelta
from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import APIKeyHeader
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as aioredis

from app.models.database import ApiKey, Tenant, UsageRecord, get_db, PlanType
from app.config import get_settings

settings = get_settings()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Redis connection for rate limiting
_redis = None


async def get_redis():
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis


def hash_api_key(key: str) -> str:
    """SHA-256 hash of API key."""
    return hashlib.sha256(key.encode()).hexdigest()


async def get_current_tenant(
    request: Request,
    api_key: str = Security(api_key_header),
    db: AsyncSession = Depends(get_db),
) -> Tenant:
    """Authenticate request via API key and return tenant."""

    if not api_key:
        raise HTTPException(status_code=401, detail="API key required. Pass via X-API-Key header.")

    # Extract prefix for fast lookup
    if api_key.startswith("dl_live_"):
        prefix = api_key[8:16]  # First 8 chars after prefix
    else:
        prefix = api_key[:8]

    key_hash = hash_api_key(api_key)

    # Find API key
    result = await db.execute(
        select(ApiKey).where(
            ApiKey.key_prefix == prefix,
            ApiKey.key_hash == key_hash,
            ApiKey.is_active == True,
        )
    )
    api_key_record = result.scalar_one_or_none()

    if not api_key_record:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Get tenant
    result = await db.execute(
        select(Tenant).where(
            Tenant.id == api_key_record.tenant_id,
            Tenant.is_active == True,
        )
    )
    tenant = result.scalar_one_or_none()

    if not tenant:
        raise HTTPException(status_code=403, detail="Tenant account is inactive")

    # Update last used
    api_key_record.last_used_at = datetime.now(timezone.utc)
    await db.commit()

    # Check rate limit
    await _check_rate_limit(tenant, db)

    return tenant


async def _check_rate_limit(tenant: Tenant, db: AsyncSession):
    """Check if tenant has exceeded their daily rate limit."""
    try:
        rd = await get_redis()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = f"ratelimit:{tenant.id}:{today}"

        current = await rd.get(key)
        count = int(current) if current else 0

        if count >= tenant.daily_limit:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "limit": tenant.daily_limit,
                    "used": count,
                    "plan": tenant.plan.value,
                    "resets_at": f"{today}T23:59:59Z",
                }
            )

        # Increment counter
        pipe = rd.pipeline()
        pipe.incr(key)
        pipe.expire(key, 86400)  # 24h TTL
        await pipe.execute()

    except aioredis.ConnectionError:
        # If Redis is down, fall back to DB check (less performant)
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        result = await db.execute(
            select(func.count()).select_from(UsageRecord).where(
                UsageRecord.tenant_id == tenant.id,
                UsageRecord.date >= today_start,
            )
        )
        # Don't block if we can't check


async def record_usage(tenant_id, processing_time_ms: int, success: bool, db: AsyncSession):
    """Record API usage for billing."""
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    result = await db.execute(
        select(UsageRecord).where(
            UsageRecord.tenant_id == tenant_id,
            UsageRecord.date == today,
        )
    )
    record = result.scalar_one_or_none()

    if record:
        record.request_count += 1
        record.total_processing_ms += processing_time_ms
        if success:
            record.successful_count += 1
        else:
            record.failed_count += 1
    else:
        record = UsageRecord(
            tenant_id=tenant_id,
            date=today,
            request_count=1,
            successful_count=1 if success else 0,
            failed_count=0 if success else 1,
            total_processing_ms=processing_time_ms,
        )
        db.add(record)

    await db.commit()
