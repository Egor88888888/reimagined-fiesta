"""
DocLens Parser: INN Certificate (Russian Taxpayer ID)
"""
import re
import numpy as np
from app.parsers.base import BaseDocumentParser
from app.core.ocr_engine import OCRResult, OCRLine
from app.core.orchestrator import FieldResult


class INNParser(BaseDocumentParser):
    """Parser for INN certificate (Свидетельство о постановке на учёт)."""

    # INN: 10 digits (legal entity) or 12 digits (individual)
    INN_RE = re.compile(r"\b(\d{12}|\d{10})\b")

    def parse(self, ocr_result: OCRResult, image: np.ndarray) -> dict[str, FieldResult]:
        fields: dict[str, FieldResult] = {}
        lines = ocr_result.lines

        # ========== INN NUMBER ==========
        # First try to find near "ИНН" label
        for i, line in enumerate(lines):
            if re.search(r"(?i)инн|идентификационн", line.text):
                # Look in same line and next 2 lines
                for j in range(max(0, i - 1), min(len(lines), i + 3)):
                    m = self.INN_RE.search(lines[j].text)
                    if m:
                        inn_value = m.group(1)
                        fields["inn_number"] = FieldResult(
                            value=inn_value,
                            confidence=lines[j].confidence,
                            source="ocr"
                        )
                        fields["inn_type"] = FieldResult(
                            value="personal" if len(inn_value) == 12 else "legal",
                            confidence=0.99,
                            source="derived"
                        )
                        break
                if "inn_number" in fields:
                    break

        # Fallback: find any 10 or 12 digit number
        if "inn_number" not in fields:
            for line in lines:
                m = self.INN_RE.search(line.text)
                if m:
                    fields["inn_number"] = FieldResult(
                        value=m.group(1),
                        confidence=line.confidence * 0.7,
                        source="ocr"
                    )
                    break

        # ========== FIO ==========
        # INN certificate typically has full name in one or multiple lines
        fio_labels = [
            (r"(?i)фамили[яю]", "last_name"),
            (r"(?i)(?<!фа)им[яю](?!\s*отч)", "first_name"),
            (r"(?i)отчеств", "patronymic"),
        ]
        for pattern, field_name in fio_labels:
            found = self.find_by_label(lines, pattern)
            if found:
                value = re.sub(pattern, "", found.text, flags=re.IGNORECASE)
                value = re.sub(r"^[:\s]+", "", value).strip()
                if value and len(value) > 1:
                    fields[field_name] = FieldResult(
                        value=value.upper(), confidence=found.confidence, source="ocr"
                    )

        # ========== BIRTH DATE ==========
        result = self.extract_date(lines, r"(?i)дат[аы]\s*рожден")
        if result:
            fields["birth_date"] = FieldResult(value=result[0], confidence=result[1], source="ocr")

        # ========== ISSUE DATE ==========
        result = self.extract_date(lines, r"(?i)дат[аы]\s*(выдач|постановк)")
        if result:
            fields["issue_date"] = FieldResult(value=result[0], confidence=result[1], source="ocr")

        # ========== ISSUING AUTHORITY ==========
        found = self.find_by_label(lines, r"(?i)(выда[нв]|налогов\w+\s*орган|инспекц)")
        if found:
            fields["issuer"] = FieldResult(
                value=self.clean_text(found.text), confidence=found.confidence, source="ocr"
            )

        return fields
