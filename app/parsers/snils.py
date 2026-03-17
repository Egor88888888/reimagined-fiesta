"""
DocLens Parser: SNILS (Russian Social Insurance Number)
"""
import re
import numpy as np
from app.parsers.base import BaseDocumentParser
from app.core.ocr_engine import OCRResult, OCRLine
from app.core.orchestrator import FieldResult


class SNILSParser(BaseDocumentParser):
    """Parser for SNILS certificate."""

    # SNILS number: XXX-XXX-XXX XX
    SNILS_RE = re.compile(r"(\d{3})[-\s]?(\d{3})[-\s]?(\d{3})[-\s]?(\d{2})")

    def parse(self, ocr_result: OCRResult, image: np.ndarray) -> dict[str, FieldResult]:
        fields: dict[str, FieldResult] = {}
        lines = ocr_result.lines

        # ========== SNILS NUMBER ==========
        for line in lines:
            m = self.SNILS_RE.search(line.text.replace(" ", "").replace("-", ""))
            if not m:
                m = self.SNILS_RE.search(line.text)
            if m:
                number = f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}"
                fields["snils_number"] = FieldResult(
                    value=number, confidence=line.confidence, source="ocr"
                )
                break

        # Also search for raw 11-digit number
        if "snils_number" not in fields:
            for line in lines:
                digits = re.sub(r"\D", "", line.text)
                if len(digits) == 11:
                    number = f"{digits[:3]}-{digits[3:6]}-{digits[6:9]} {digits[9:]}"
                    fields["snils_number"] = FieldResult(
                        value=number, confidence=line.confidence * 0.8, source="ocr"
                    )
                    break

        # ========== FIO ==========
        # SNILS typically has: Фамилия, Имя, Отчество on separate lines
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

        # ========== BIRTH PLACE ==========
        found = self.find_by_label(lines, r"(?i)место\s*рожден")
        if found:
            value = re.sub(r"(?i)место\s*рождения\s*:?\s*", "", found.text).strip()
            if value:
                fields["birth_place"] = FieldResult(
                    value=value, confidence=found.confidence, source="ocr"
                )

        # ========== SEX ==========
        for line in lines:
            if re.search(r"(?i)\bмужской\b|\bмуж\b", line.text):
                fields["sex"] = FieldResult(value="М", confidence=line.confidence, source="ocr")
                break
            if re.search(r"(?i)\bженский\b|\bжен\b", line.text):
                fields["sex"] = FieldResult(value="Ж", confidence=line.confidence, source="ocr")
                break

        # ========== REGISTRATION DATE ==========
        result = self.extract_date(lines, r"(?i)дат[аы]\s*регистрац")
        if result:
            fields["registration_date"] = FieldResult(
                value=result[0], confidence=result[1], source="ocr"
            )

        return fields
