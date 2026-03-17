"""
DocLens ML Corrector
Post-processes OCR output using learned patterns and a neural model.

Two-tier correction:
1. Rule-based: applies known CorrectionPatterns from DB (fast, deterministic)
2. ML-based: sequence-to-sequence model for harder corrections (optional)

The rule-based tier works immediately from user feedback.
The ML tier requires training (see trainer.py).
"""
import re
import json
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.models import CorrectionPattern

logger = logging.getLogger(__name__)

# Directory for cached model artifacts
MODEL_DIR = Path("/app/ml_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Field Validation Rules (domain knowledge for Russian documents)
# ============================================================

FIELD_VALIDATORS = {
    "passport_rf": {
        "series": {
            "pattern": r"^\d{2}\s\d{2}$",
            "description": "XX XX (4 цифры через пробел)",
        },
        "number": {
            "pattern": r"^\d{6}$",
            "description": "6 цифр",
        },
        "department_code": {
            "pattern": r"^\d{3}-\d{3}$",
            "description": "XXX-XXX",
        },
        "issue_date": {
            "pattern": r"^\d{2}\.\d{2}\.\d{4}$",
            "description": "ДД.ММ.ГГГГ",
        },
        "birth_date": {
            "pattern": r"^\d{2}\.\d{2}\.\d{4}$",
            "description": "ДД.ММ.ГГГГ",
        },
        "sex": {
            "pattern": r"^[МЖ]$",
            "description": "М или Ж",
        },
        "last_name": {
            "pattern": r"^[А-ЯЁ][а-яё\-]+$",
            "description": "Кириллица, начинается с заглавной",
            "transform": "capitalize_cyrillic",
        },
        "first_name": {
            "pattern": r"^[А-ЯЁ][а-яё\-]+$",
            "description": "Кириллица, начинается с заглавной",
            "transform": "capitalize_cyrillic",
        },
        "patronymic": {
            "pattern": r"^[А-ЯЁ][а-яё\-]+$",
            "description": "Кириллица, начинается с заглавной",
            "transform": "capitalize_cyrillic",
        },
    },
    "snils": {
        "number": {
            "pattern": r"^\d{3}-\d{3}-\d{3}\s\d{2}$",
            "description": "XXX-XXX-XXX XX",
        },
    },
    "inn": {
        "number": {
            "pattern": r"^\d{10,12}$",
            "description": "10 или 12 цифр",
        },
    },
}

# Common OCR errors for Russian text
CYRILLIC_OCR_FIXES = {
    # Цифры ↔ Кириллица
    "0": "О", "O": "О",
    "3": "З", "Z": "З",
    "6": "б",
    "8": "В",
    # Латиница ↔ Кириллица
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н",
    "K": "К", "M": "М", "O": "О", "P": "Р", "T": "Т",
    "X": "Х", "a": "а", "c": "с", "e": "е", "o": "о",
    "p": "р", "x": "х", "y": "у",
}

# Reverse map for fields that should contain digits
CYRILLIC_TO_DIGIT_FIXES = {
    "О": "0", "о": "0",
    "З": "3", "з": "3",
    "б": "6", "Б": "6",
    "В": "8",
    "l": "1", "I": "1",
    "S": "5", "s": "5",
}


class FieldCorrector:
    """Corrects OCR-extracted fields using patterns and ML."""

    def __init__(self):
        self._patterns_cache: dict[str, list[CorrectionPattern]] = {}
        self._patterns_loaded = False
        self._ml_model = None
        self._ml_model_loaded = False

    async def load_patterns(self, db: AsyncSession):
        """Load correction patterns from DB into memory cache."""
        result = await db.execute(
            select(CorrectionPattern).where(CorrectionPattern.is_active == True)
        )
        patterns = result.scalars().all()

        self._patterns_cache.clear()
        for p in patterns:
            key = f"{p.document_type}:{p.field_name}"
            if key not in self._patterns_cache:
                self._patterns_cache[key] = []
            self._patterns_cache[key].append(p)

        self._patterns_loaded = True
        logger.info(f"Loaded {len(patterns)} correction patterns into cache")

    def load_ml_model(self, model_path: str = None):
        """Load the trained ML corrector model.

        Uses a lightweight character-level seq2seq model
        or a fine-tuned transformer depending on training.
        """
        if model_path is None:
            model_path = str(MODEL_DIR / "corrector_latest.npz")

        path = Path(model_path)
        if not path.exists():
            logger.info("No ML model found, using rule-based correction only")
            self._ml_model = None
            return

        try:
            data = np.load(model_path, allow_pickle=True)
            self._ml_model = {
                "char_bigram_probs": data.get("char_bigram_probs"),
                "field_error_probs": data.get("field_error_probs"),
                "substitution_matrix": data.get("substitution_matrix"),
                "vocab": data.get("vocab", None),
                "metadata": json.loads(str(data.get("metadata", "{}")))
            }
            self._ml_model_loaded = True
            logger.info(f"Loaded ML corrector model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            self._ml_model = None

    async def correct_fields(
        self,
        db: AsyncSession,
        document_type: str,
        fields: dict,
    ) -> dict:
        """Apply corrections to all extracted fields.

        Args:
            document_type: e.g. "passport_rf"
            fields: {field_name: {"value": str, "confidence": float, ...}}

        Returns:
            Updated fields dict with corrections applied.
        """
        if not self._patterns_loaded:
            await self.load_patterns(db)

        corrected_fields = {}
        for field_name, field_data in fields.items():
            if not isinstance(field_data, dict) or "value" not in field_data:
                corrected_fields[field_name] = field_data
                continue

            original_value = field_data["value"]
            confidence = field_data.get("confidence", 0.0)

            # Apply corrections in order of reliability
            corrected_value = original_value
            correction_source = None

            # Tier 1: Domain-specific format fixes
            format_result = self._apply_format_correction(
                document_type, field_name, corrected_value
            )
            if format_result != corrected_value:
                corrected_value = format_result
                correction_source = "format_rules"

            # Tier 2: Known OCR character substitution fixes
            char_result = self._apply_char_corrections(
                document_type, field_name, corrected_value
            )
            if char_result != corrected_value:
                corrected_value = char_result
                correction_source = "char_correction"

            # Tier 3: Learned patterns from user feedback
            pattern_result = self._apply_pattern_corrections(
                document_type, field_name, corrected_value
            )
            if pattern_result != corrected_value:
                corrected_value = pattern_result
                correction_source = "learned_pattern"

            # Tier 4: ML model correction (if available)
            if self._ml_model and confidence < 0.9:
                ml_result = self._apply_ml_correction(
                    document_type, field_name, corrected_value
                )
                if ml_result != corrected_value:
                    corrected_value = ml_result
                    correction_source = "ml_model"

            # Build corrected field
            corrected_field = dict(field_data)
            if corrected_value != original_value:
                corrected_field["value"] = corrected_value
                corrected_field["original_ocr_value"] = original_value
                corrected_field["correction_source"] = correction_source
                # Boost confidence slightly for corrected values
                corrected_field["confidence"] = min(
                    0.99,
                    confidence + 0.05
                )
                logger.debug(
                    f"Corrected {field_name}: "
                    f"'{original_value}' -> '{corrected_value}' "
                    f"(source: {correction_source})"
                )

            corrected_fields[field_name] = corrected_field

        return corrected_fields

    def _apply_format_correction(
        self,
        document_type: str,
        field_name: str,
        value: str,
    ) -> str:
        """Apply format-based corrections using domain knowledge."""
        validators = FIELD_VALIDATORS.get(document_type, {})
        field_rules = validators.get(field_name, {})

        if not field_rules:
            return value

        transform = field_rules.get("transform")
        if transform == "capitalize_cyrillic":
            value = self._capitalize_cyrillic(value)

        # Fix department code format
        if field_name == "department_code" and document_type == "passport_rf":
            digits = re.sub(r"\D", "", value)
            if len(digits) == 6:
                value = f"{digits[:3]}-{digits[3:]}"

        # Fix SNILS format
        if field_name == "number" and document_type == "snils":
            digits = re.sub(r"\D", "", value)
            if len(digits) == 11:
                value = f"{digits[:3]}-{digits[3:6]}-{digits[6:9]} {digits[9:]}"

        # Fix date format
        if field_name in ("issue_date", "birth_date"):
            # Normalize various separators to dots
            value = re.sub(r"[\-/]", ".", value)
            # Fix common date OCR issues
            m = re.match(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", value)
            if m:
                day = m.group(1).zfill(2)
                month = m.group(2).zfill(2)
                year = m.group(3)
                value = f"{day}.{month}.{year}"

        return value

    def _apply_char_corrections(
        self,
        document_type: str,
        field_name: str,
        value: str,
    ) -> str:
        """Apply character-level OCR error corrections."""
        validators = FIELD_VALIDATORS.get(document_type, {})
        field_rules = validators.get(field_name, {})
        pattern = field_rules.get("pattern", "")

        # Determine if field should be digits or Cyrillic
        expects_digits = bool(re.search(r"\\d", pattern))
        expects_cyrillic = bool(re.search(r"\[А-Я", pattern))

        if expects_digits:
            # Replace Cyrillic lookalikes with digits
            corrected = []
            for ch in value:
                corrected.append(CYRILLIC_TO_DIGIT_FIXES.get(ch, ch))
            return "".join(corrected)

        if expects_cyrillic:
            # Replace Latin/digit lookalikes with Cyrillic
            corrected = []
            for ch in value:
                if ch in CYRILLIC_OCR_FIXES and not ch.isdigit():
                    corrected.append(CYRILLIC_OCR_FIXES[ch])
                else:
                    corrected.append(ch)
            return "".join(corrected)

        return value

    def _apply_pattern_corrections(
        self,
        document_type: str,
        field_name: str,
        value: str,
    ) -> str:
        """Apply corrections from learned patterns (user feedback)."""
        key = f"{document_type}:{field_name}"
        patterns = self._patterns_cache.get(key, [])

        if not patterns:
            return value

        # Sort by confidence (highest first)
        patterns.sort(key=lambda p: p.confidence, reverse=True)

        for pattern in patterns:
            if pattern.confidence < 0.6:
                continue

            if pattern.pattern_type == "char_substitution":
                if pattern.error_pattern in value:
                    value = value.replace(
                        pattern.error_pattern,
                        pattern.correction
                    )

            elif pattern.pattern_type == "format":
                if value == pattern.error_pattern:
                    value = pattern.correction

        return value

    def _apply_ml_correction(
        self,
        document_type: str,
        field_name: str,
        value: str,
    ) -> str:
        """Apply ML model-based correction.

        Uses character bigram probabilities and a substitution matrix
        trained on accumulated correction data.
        """
        if not self._ml_model:
            return value

        try:
            sub_matrix = self._ml_model.get("substitution_matrix")
            vocab = self._ml_model.get("vocab")

            if sub_matrix is None or vocab is None:
                return value

            # Convert vocab to lookup dict
            if isinstance(vocab, np.ndarray):
                vocab = vocab.item() if vocab.ndim == 0 else dict(enumerate(vocab))

            # Simple approach: for each character, check if substitution
            # has higher probability than keeping original
            corrected = []
            for ch in value:
                if ch in vocab:
                    ch_idx = vocab[ch]
                    row = sub_matrix[ch_idx]
                    best_idx = np.argmax(row)
                    best_prob = row[best_idx]

                    # Only substitute if ML is confident (>0.7)
                    # and original char probability is low (<0.5)
                    orig_prob = row[ch_idx] if ch_idx < len(row) else 0
                    if best_prob > 0.7 and orig_prob < 0.5 and best_idx != ch_idx:
                        # Find character for best_idx
                        reverse_vocab = {v: k for k, v in vocab.items()}
                        if best_idx in reverse_vocab:
                            corrected.append(reverse_vocab[best_idx])
                            continue

                corrected.append(ch)

            return "".join(corrected)

        except Exception as e:
            logger.warning(f"ML correction failed for {field_name}: {e}")
            return value

    def _capitalize_cyrillic(self, value: str) -> str:
        """Capitalize Cyrillic name properly: ИВАНОВ → Иванов."""
        if not value:
            return value

        # If all caps, convert to title case
        if value == value.upper() and len(value) > 1:
            # Handle hyphenated names
            parts = value.split("-")
            return "-".join(p.capitalize() for p in parts)

        return value

    def validate_field(
        self,
        document_type: str,
        field_name: str,
        value: str,
    ) -> dict:
        """Validate a field value against known format rules.

        Returns:
            {"valid": bool, "message": str, "expected_format": str}
        """
        validators = FIELD_VALIDATORS.get(document_type, {})
        field_rules = validators.get(field_name, {})

        if not field_rules:
            return {"valid": True, "message": "no validation rules"}

        pattern = field_rules.get("pattern")
        if pattern and not re.match(pattern, value):
            return {
                "valid": False,
                "message": f"Не соответствует формату: {field_rules.get('description', pattern)}",
                "expected_format": field_rules.get("description", pattern),
            }

        return {"valid": True, "message": "ok"}


# Singleton
_corrector = None


def get_corrector() -> FieldCorrector:
    global _corrector
    if _corrector is None:
        _corrector = FieldCorrector()
    return _corrector
