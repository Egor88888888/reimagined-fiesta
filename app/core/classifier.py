"""
DocLens Document Classifier
Determines document type from OCR text (rule-based).
"""
import re
import logging
from app.models.database import DocumentType
from app.core.ocr_engine import OCRResult

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """Rule-based document type classifier."""

    # Patterns for each document type
    PATTERNS = {
        DocumentType.PASSPORT_RF: [
            r"(?i)паспорт\s+гражданина\s+российской\s+федерации",
            r"(?i)российская\s+федерация",
            r"(?i)министерство\s+внутренних\s+дел",
            r"(?i)отдел[а-я]*\s+(УФМС|УМВД|МВД|полиции)",
            r"\d{2}\s?\d{2}\s?\d{6}",  # Series + number pattern
        ],
        DocumentType.PASSPORT_CIS: [
            r"(?i)passeport|passport|паспорт",
            r"(?i)(O.?ZBEKISTON|QIRG.?IZSTAN|ТОҶИКИСТОН|ҚАЗАҚСТАН|КАЗАХСТАН)",
            r"(?i)(РЕСПУБЛИКА|REPUBLIC|RESPUBLIKASI)",
            r"[A-Z<]{2}[A-Z<]{3}[A-Z<]+",  # MRZ pattern
        ],
        DocumentType.DRIVER_LICENSE: [
            r"(?i)водительское\s+удостоверени[ея]",
            r"(?i)driving\s+licen[sc]e",
            r"(?i)permis\s+de\s+conduire",
            r"(?i)категор[а-я]+\s*[ABCDEM]",
            r"(?i)ГИБДД|ГАИ",
        ],
        DocumentType.SNILS: [
            r"\d{3}[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{2}",  # XXX-XXX-XXX XX
            r"(?i)страховое\s+свидетельство",
            r"(?i)пенсионного\s+страхования",
            r"(?i)СНИЛС",
        ],
        DocumentType.INN: [
            r"(?i)свидетельство\s+о\s+постановке",
            r"(?i)учет\s+в\s+налоговом\s+органе",
            r"(?i)идентификационный\s+номер\s+налогоплательщика",
            r"(?i)ИНН\s*:?\s*\d{10,12}",
            r"(?i)федеральная\s+налоговая\s+служба",
        ],
    }

    # Scoring weights
    WEIGHTS = {
        DocumentType.PASSPORT_RF: [5, 3, 3, 4, 2],
        DocumentType.PASSPORT_CIS: [2, 5, 3, 4],
        DocumentType.DRIVER_LICENSE: [5, 4, 3, 3, 4],
        DocumentType.SNILS: [5, 4, 4, 5],
        DocumentType.INN: [4, 4, 4, 5, 3],
    }

    def classify(self, ocr_result: OCRResult, hint: str = None) -> tuple[DocumentType, float]:
        """Classify document type based on OCR text.

        Args:
            ocr_result: OCR extraction result
            hint: Optional type hint from client

        Returns:
            Tuple of (DocumentType, confidence 0.0-1.0)
        """
        # If hint provided and valid, boost its score
        hint_type = None
        if hint:
            try:
                hint_type = DocumentType(hint)
            except ValueError:
                pass

        full_text = ocr_result.full_text
        scores: dict[DocumentType, float] = {}

        for doc_type, patterns in self.PATTERNS.items():
            weights = self.WEIGHTS[doc_type]
            total_weight = sum(weights)
            score = 0.0

            for i, pattern in enumerate(patterns):
                if re.search(pattern, full_text):
                    score += weights[i]

            # Normalize to 0-1
            scores[doc_type] = score / total_weight if total_weight > 0 else 0

            # Boost hinted type
            if hint_type and doc_type == hint_type:
                scores[doc_type] = min(1.0, scores[doc_type] + 0.2)

        # MRZ detection boosts passport types
        if self._has_mrz(full_text):
            for t in [DocumentType.PASSPORT_RF, DocumentType.PASSPORT_CIS]:
                scores[t] = min(1.0, scores.get(t, 0) + 0.15)

        if not scores:
            return DocumentType.UNKNOWN, 0.0

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        # If best score is too low, mark as unknown
        if best_score < 0.3:
            return DocumentType.UNKNOWN, best_score

        logger.info(f"Classified as {best_type.value} (confidence={best_score:.2f}), scores={scores}")
        return best_type, best_score

    def _has_mrz(self, text: str) -> bool:
        """Check if text contains MRZ lines."""
        mrz_pattern = r"[A-Z0-9<]{30,44}"
        lines = text.split("\n")
        mrz_lines = [l for l in lines if re.match(mrz_pattern, l.replace(" ", ""))]
        return len(mrz_lines) >= 2
