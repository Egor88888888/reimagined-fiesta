"""
DocLens Feedback Collector
Handles user corrections and converts them into training data.

Active learning loop:
1. User corrects an OCR field → FieldCorrection saved
2. Corrections are periodically aggregated into CorrectionPatterns
3. Patterns are used immediately for rule-based fixes
4. Accumulated data trains the ML corrector model
"""
import logging
import hashlib
from datetime import datetime, timezone
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.models import (
    FieldCorrection, CorrectionPattern, CorrectionStatus
)

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """Collects and processes user feedback for active learning."""

    async def submit_correction(
        self,
        db: AsyncSession,
        recognition_id: str,
        tenant_id: str,
        document_type: str,
        field_name: str,
        original_value: str,
        corrected_value: str,
        original_confidence: float = 0.0,
        ocr_raw_text: str = None,
    ) -> FieldCorrection:
        """Submit a single field correction from the user.

        Auto-approves corrections that match known patterns.
        """
        # Skip if nothing changed
        if original_value.strip() == corrected_value.strip():
            return None

        # Create image region hash for dedup
        region_hash = hashlib.sha256(
            f"{document_type}:{field_name}:{original_value}".encode()
        ).hexdigest()[:16]

        # Check for duplicate correction
        existing = await db.execute(
            select(FieldCorrection).where(
                FieldCorrection.recognition_id == recognition_id,
                FieldCorrection.field_name == field_name,
            )
        )
        if existing.scalar_one_or_none():
            # Update existing correction
            await db.execute(
                update(FieldCorrection)
                .where(
                    FieldCorrection.recognition_id == recognition_id,
                    FieldCorrection.field_name == field_name,
                )
                .values(
                    corrected_value=corrected_value,
                    status=CorrectionStatus.PENDING,
                )
            )
            await db.flush()
            result = await db.execute(
                select(FieldCorrection).where(
                    FieldCorrection.recognition_id == recognition_id,
                    FieldCorrection.field_name == field_name,
                )
            )
            return result.scalar_one()

        # Auto-approve if we've seen this exact correction 3+ times
        status = CorrectionStatus.PENDING
        pattern_count = await self._count_matching_corrections(
            db, document_type, field_name, original_value, corrected_value
        )
        if pattern_count >= 3:
            status = CorrectionStatus.APPROVED
            logger.info(
                f"Auto-approved correction: {field_name} "
                f"'{original_value}' -> '{corrected_value}' "
                f"(seen {pattern_count} times)"
            )

        correction = FieldCorrection(
            recognition_id=recognition_id,
            tenant_id=tenant_id,
            document_type=document_type,
            field_name=field_name,
            original_value=original_value,
            corrected_value=corrected_value,
            original_confidence=original_confidence,
            ocr_raw_text=ocr_raw_text,
            image_region_hash=region_hash,
            status=status,
        )
        db.add(correction)
        await db.flush()

        # Trigger pattern extraction if enough corrections accumulated
        pending_count = await self._count_pending(db, document_type)
        if pending_count >= 10:
            await self.extract_patterns(db, document_type)

        return correction

    async def submit_bulk_corrections(
        self,
        db: AsyncSession,
        recognition_id: str,
        tenant_id: str,
        document_type: str,
        corrections: dict[str, dict],
    ) -> list[FieldCorrection]:
        """Submit corrections for multiple fields at once.

        Args:
            corrections: {field_name: {"original": str, "corrected": str, "confidence": float}}
        """
        results = []
        for field_name, data in corrections.items():
            correction = await self.submit_correction(
                db=db,
                recognition_id=recognition_id,
                tenant_id=tenant_id,
                document_type=document_type,
                field_name=field_name,
                original_value=data.get("original", ""),
                corrected_value=data.get("corrected", ""),
                original_confidence=data.get("confidence", 0.0),
            )
            if correction:
                results.append(correction)
        return results

    async def extract_patterns(
        self,
        db: AsyncSession,
        document_type: str = None,
    ) -> list[CorrectionPattern]:
        """Analyze approved corrections and extract error patterns.

        Detects:
        - Character substitutions (e.g., "Б" → "6", "О" → "0")
        - Format errors (e.g., missing hyphens in codes)
        - Systematic errors per field type
        """
        query = select(FieldCorrection).where(
            FieldCorrection.status.in_([
                CorrectionStatus.APPROVED,
                CorrectionStatus.PENDING,
            ])
        )
        if document_type:
            query = query.where(FieldCorrection.document_type == document_type)

        result = await db.execute(query)
        corrections = result.scalars().all()

        if not corrections:
            return []

        # Group by (document_type, field_name)
        grouped = defaultdict(list)
        for c in corrections:
            grouped[(c.document_type, c.field_name)].append(c)

        new_patterns = []

        for (doc_type, field_name), field_corrections in grouped.items():
            # Extract character-level substitution patterns
            char_patterns = self._extract_char_substitutions(field_corrections)
            for error_char, correct_char, count in char_patterns:
                pattern = await self._upsert_pattern(
                    db,
                    doc_type=doc_type,
                    field_name=field_name,
                    error_pattern=error_char,
                    correction=correct_char,
                    pattern_type="char_substitution",
                    count=count,
                )
                new_patterns.append(pattern)

            # Extract format patterns
            format_patterns = self._extract_format_patterns(field_corrections)
            for error_fmt, correct_fmt, count in format_patterns:
                pattern = await self._upsert_pattern(
                    db,
                    doc_type=doc_type,
                    field_name=field_name,
                    error_pattern=error_fmt,
                    correction=correct_fmt,
                    pattern_type="format",
                    count=count,
                )
                new_patterns.append(pattern)

        await db.flush()
        logger.info(f"Extracted {len(new_patterns)} patterns from {len(corrections)} corrections")
        return new_patterns

    def _extract_char_substitutions(
        self,
        corrections: list[FieldCorrection],
    ) -> list[tuple[str, str, int]]:
        """Find common character-level substitution errors."""
        char_subs = Counter()

        for c in corrections:
            orig = c.original_value
            corr = c.corrected_value

            # Use SequenceMatcher to find character-level diffs
            matcher = SequenceMatcher(None, orig, corr)
            for op, i1, i2, j1, j2 in matcher.get_opcodes():
                if op == "replace":
                    old_chunk = orig[i1:i2]
                    new_chunk = corr[j1:j2]
                    # Only single-char or short substitutions
                    if len(old_chunk) <= 3 and len(new_chunk) <= 3:
                        char_subs[(old_chunk, new_chunk)] += 1

        # Return patterns that appeared 2+ times
        return [
            (error, correct, count)
            for (error, correct), count in char_subs.most_common()
            if count >= 2
        ]

    def _extract_format_patterns(
        self,
        corrections: list[FieldCorrection],
    ) -> list[tuple[str, str, int]]:
        """Find common format correction patterns."""
        format_subs = Counter()

        for c in corrections:
            orig = c.original_value.strip()
            corr = c.corrected_value.strip()

            # Detect formatting-only changes (same alphanumeric content)
            orig_alpha = "".join(ch for ch in orig if ch.isalnum())
            corr_alpha = "".join(ch for ch in corr if ch.isalnum())

            if orig_alpha == corr_alpha and orig != corr:
                # Pure formatting change
                format_subs[(orig, corr)] += 1

        return [
            (error, correct, count)
            for (error, correct), count in format_subs.most_common()
            if count >= 2
        ]

    async def _upsert_pattern(
        self,
        db: AsyncSession,
        doc_type: str,
        field_name: str,
        error_pattern: str,
        correction: str,
        pattern_type: str,
        count: int,
    ) -> CorrectionPattern:
        """Create or update a correction pattern."""
        result = await db.execute(
            select(CorrectionPattern).where(
                CorrectionPattern.document_type == doc_type,
                CorrectionPattern.field_name == field_name,
                CorrectionPattern.error_pattern == error_pattern,
                CorrectionPattern.correction == correction,
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            existing.occurrence_count += count
            existing.confidence = min(0.99, 0.5 + (existing.occurrence_count * 0.05))
            existing.last_seen_at = datetime.now(timezone.utc)
            return existing

        pattern = CorrectionPattern(
            document_type=doc_type,
            field_name=field_name,
            error_pattern=error_pattern,
            correction=correction,
            pattern_type=pattern_type,
            occurrence_count=count,
            confidence=min(0.99, 0.5 + (count * 0.05)),
            last_seen_at=datetime.now(timezone.utc),
        )
        db.add(pattern)
        return pattern

    async def _count_matching_corrections(
        self,
        db: AsyncSession,
        document_type: str,
        field_name: str,
        original_value: str,
        corrected_value: str,
    ) -> int:
        """Count how many times this exact correction has been seen."""
        result = await db.execute(
            select(func.count()).select_from(FieldCorrection).where(
                FieldCorrection.document_type == document_type,
                FieldCorrection.field_name == field_name,
                FieldCorrection.original_value == original_value,
                FieldCorrection.corrected_value == corrected_value,
                FieldCorrection.status.in_([
                    CorrectionStatus.APPROVED,
                    CorrectionStatus.USED_IN_TRAINING,
                ]),
            )
        )
        return result.scalar() or 0

    async def _count_pending(self, db: AsyncSession, document_type: str) -> int:
        """Count pending corrections for a document type."""
        result = await db.execute(
            select(func.count()).select_from(FieldCorrection).where(
                FieldCorrection.document_type == document_type,
                FieldCorrection.status == CorrectionStatus.PENDING,
            )
        )
        return result.scalar() or 0

    async def get_correction_stats(
        self,
        db: AsyncSession,
        document_type: str = None,
    ) -> dict:
        """Get statistics about corrections and patterns."""
        query = select(
            FieldCorrection.document_type,
            FieldCorrection.field_name,
            FieldCorrection.status,
            func.count().label("count"),
        ).group_by(
            FieldCorrection.document_type,
            FieldCorrection.field_name,
            FieldCorrection.status,
        )
        if document_type:
            query = query.where(FieldCorrection.document_type == document_type)

        result = await db.execute(query)
        rows = result.all()

        stats = defaultdict(lambda: defaultdict(dict))
        for row in rows:
            stats[row.document_type][row.field_name][row.status.value] = row.count

        # Pattern counts
        pattern_query = select(
            CorrectionPattern.document_type,
            func.count().label("count"),
        ).where(
            CorrectionPattern.is_active == True
        ).group_by(CorrectionPattern.document_type)

        if document_type:
            pattern_query = pattern_query.where(
                CorrectionPattern.document_type == document_type
            )

        pattern_result = await db.execute(pattern_query)
        pattern_counts = {row.document_type: row.count for row in pattern_result.all()}

        return {
            "corrections": dict(stats),
            "active_patterns": pattern_counts,
        }


# Singleton
_feedback_collector = None


def get_feedback_collector() -> FeedbackCollector:
    global _feedback_collector
    if _feedback_collector is None:
        _feedback_collector = FeedbackCollector()
    return _feedback_collector
