"""
DocLens Parser: Russian Driver's License
"""
import re
import numpy as np
from app.parsers.base import BaseDocumentParser
from app.core.ocr_engine import OCRResult, OCRLine
from app.core.orchestrator import FieldResult


class DriverLicenseParser(BaseDocumentParser):
    """Parser for Russian driver's license (new format)."""

    CATEGORIES_RE = re.compile(r"[ABCDEM][12]?[E]?")

    def parse(self, ocr_result: OCRResult, image: np.ndarray) -> dict[str, FieldResult]:
        fields: dict[str, FieldResult] = {}
        lines = ocr_result.lines

        # ========== LAST NAME ==========
        found = self.find_by_label(lines, r"(?i)(фамили[яю]|1\.?\s*surname|1\.)")
        if found:
            value = self._clean_label(found.text, r"(?i)(фамили[яю]|1\.?\s*surname|1\.)")
            if value:
                fields["last_name"] = FieldResult(value=value.upper(), confidence=found.confidence, source="ocr")

        # ========== FIRST NAME + PATRONYMIC ==========
        found = self.find_by_label(lines, r"(?i)(имя|2\.?\s*name|2\.)")
        if found:
            value = self._clean_label(found.text, r"(?i)(имя|2\.?\s*name|2\.)")
            if value:
                parts = value.split()
                if parts:
                    fields["first_name"] = FieldResult(
                        value=parts[0].upper(), confidence=found.confidence, source="ocr"
                    )
                if len(parts) > 1:
                    fields["patronymic"] = FieldResult(
                        value=" ".join(parts[1:]).upper(), confidence=found.confidence, source="ocr"
                    )

        # ========== BIRTH DATE ==========
        found = self.find_by_label(lines, r"(?i)(дата\s*рожд|3\.?\s*date\s*of\s*birth|3\.)")
        if found:
            date_match = re.search(r"(\d{2})[.\-/](\d{2})[.\-/](\d{4})", found.text)
            if date_match:
                fields["birth_date"] = FieldResult(
                    value=f"{date_match.group(1)}.{date_match.group(2)}.{date_match.group(3)}",
                    confidence=found.confidence, source="ocr"
                )

        # ========== ISSUE DATE ==========
        result = self.extract_date(lines, r"(?i)(дата\s*выдач|4[aа]\.?)")
        if result:
            fields["issue_date"] = FieldResult(value=result[0], confidence=result[1], source="ocr")

        # ========== EXPIRY DATE ==========
        result = self.extract_date(lines, r"(?i)(действительн|4[bб]\.?|срок)")
        if result:
            fields["expiry_date"] = FieldResult(value=result[0], confidence=result[1], source="ocr")

        # ========== LICENSE NUMBER ==========
        for line in lines:
            # Format: XX XX XXXXXX or XX XX NNNNNN
            m = re.search(r"(\d{2})\s*(\d{2})\s*(\d{6})", line.text)
            if m and "number" not in fields:
                fields["number"] = FieldResult(
                    value=f"{m.group(1)} {m.group(2)} {m.group(3)}",
                    confidence=line.confidence, source="ocr"
                )

        # ========== CATEGORIES ==========
        categories = set()
        for line in lines:
            for m in self.CATEGORIES_RE.finditer(line.text):
                cat = m.group(0)
                if cat not in ("E",):  # Skip standalone E
                    categories.add(cat)
        if categories:
            fields["categories"] = FieldResult(
                value=", ".join(sorted(categories)),
                confidence=0.85, source="ocr"
            )

        # ========== ISSUER ==========
        found = self.find_by_label(lines, r"(?i)(ГИБДД|ГАИ|выда[нв])")
        if found:
            fields["issuer"] = FieldResult(
                value=self.clean_text(found.text), confidence=found.confidence, source="ocr"
            )

        return fields

    def _clean_label(self, text: str, label_pattern: str) -> str:
        """Remove label prefix from text."""
        text = re.sub(label_pattern, "", text, flags=re.IGNORECASE)
        text = re.sub(r"^[:\s/.\-]+", "", text)
        return self.clean_text(text)
