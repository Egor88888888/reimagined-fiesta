#!/usr/bin/env python3
"""
Create an admin tenant with API key.
Usage: python scripts/create_admin.py --name "Admin" --email "admin@example.com"
"""
import asyncio
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.database import Tenant, ApiKey, PlanType, init_db, async_session
from app.api.auth import hash_api_key


async def create_admin(name: str, email: str, plan: str = "enterprise"):
    await init_db()

    async with async_session() as db:
        # Check if exists
        from sqlalchemy import select
        result = await db.execute(select(Tenant).where(Tenant.email == email))
        existing = result.scalar_one_or_none()

        if existing:
            print(f"Tenant with email {email} already exists (id: {existing.id})")
            return

        # Create tenant
        plan_type = PlanType(plan)
        tenant = Tenant(name=name, email=email, plan=plan_type)
        db.add(tenant)
        await db.flush()

        # Create API key
        raw_key = ApiKey.generate_key()
        prefix = raw_key[8:16]
        key_hash = hash_api_key(raw_key)

        api_key = ApiKey(
            tenant_id=tenant.id,
            key_prefix=prefix,
            key_hash=key_hash,
            name="Admin Key",
        )
        db.add(api_key)
        await db.commit()

        print("=" * 60)
        print(f"Tenant created: {name} ({email})")
        print(f"Plan: {plan}")
        print(f"Tenant ID: {tenant.id}")
        print(f"")
        print(f"API Key (save it — shown only once!):")
        print(f"  {raw_key}")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create admin tenant")
    parser.add_argument("--name", required=True, help="Tenant name")
    parser.add_argument("--email", required=True, help="Tenant email")
    parser.add_argument("--plan", default="enterprise", choices=["free", "basic", "pro", "enterprise"])
    args = parser.parse_args()

    asyncio.run(create_admin(args.name, args.email, args.plan))
