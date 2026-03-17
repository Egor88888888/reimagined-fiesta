"""
DocLens Parser: Russian Federation Internal Passport
Pages 2-3 (main spread): personal data + photo
"""
import re
import numpy as np
from app.parsers.base import BaseDocumentParser
from app.core.ocr_engine import OCRResult, OCRLine
from app.core.orchestrator import FieldResult


class PassportRFParser(BaseDocumentParser):
    """Parser for Russian internal passport (pages 2-3)."""

    # Series/number: XX XX XXXXXX
    SERIES_NUMBER_RE = re.compile(r"(\d{2})\s*(\d{2})\s*[^\d]?\s*(\d{6})")
    # Code: XXX-XXX
    CODE_RE = re.compile(r"(\d{3})\s*[-–—.]\s*(\d{3})")
    # Date: DD.MM.YYYY
    DATE_RE = re.compile(r"(\d{2})[.\-/](\d{2})[.\-/](\d{4})")

    def parse(self, ocr_result: OCRResult, image: np.ndarray) -> dict[str, FieldResult]:
        fields: dict[str, FieldResult] = {}
        lines = ocr_result.lines

        # ========== SERIES & NUMBER ==========
        self._extract_series_number(lines, fields)

        # ========== ISSUER (кем выдан) ==========
        self._extract_issuer(lines, fields)

        # ========== ISSUE DATE (дата выдачи) ==========
        self._extract_issue_date(lines, fields)

        # ========== DEPARTMENT CODE (код подразделения) ==========
        self._extract_department_code(lines, fields)

        # ========== LAST NAME (фамилия) ==========
        self._extract_fio(lines, fields)

        # ========== BIRTH DATE ==========
        self._extract_birth_date(lines, fields)

        # ========== BIRTH PLACE ==========
        self._extract_birth_place(lines, fields)

        # ========== SEX ==========
        self._extract_sex(lines, fields)

        return fields

    def _extract_series_number(self, lines: list[OCRLine], fields: dict):
        """Extract passport series and number."""
        for line in lines:
            text = line.text.replace(" ", "").replace(".", "")
            # Look for 10-digit sequence
            m = re.search(r"(\d{4})(\d{6})", text)
            if m:
                series = f"{m.group(1)[:2]} {m.group(1)[2:]}"
                number = m.group(2)
                fields["series"] = FieldResult(
                    value=series, confidence=line.confidence, source="ocr"
                )
                fields["number"] = FieldResult(
                    value=number, confidence=line.confidence, source="ocr"
                )
                return

        # Fallback: look with original spacing
        for line in lines:
            m = self.SERIES_NUMBER_RE.search(line.text)
            if m:
                fields["series"] = FieldResult(
                    value=f"{m.group(1)} {m.group(2)}",
                    confidence=line.confidence, source="ocr"
                )
                fields["number"] = FieldResult(
                    value=m.group(3),
                    confidence=line.confidence, source="ocr"
                )
                return

    def _extract_issuer(self, lines: list[OCRLine], fields: dict):
        """Extract who issued the passport."""
        issuer_parts = []
        found = False
        for i, line in enumerate(lines):
            text_lower = line.text.lower()
            if any(kw in text_lower for kw in [
                "выдан", "отдел", "уфмс", "умвд", "мвд", "полиции",
                "управлени", "министерств"
            ]):
                found = True
                # Collect this and next 1-2 lines as issuer
                issuer_parts.append(line.text)
                for j in range(1, 3):
                    if i + j < len(lines):
                        next_line = lines[i + j]
                        # Stop if we hit a date or code
                        if self.DATE_RE.search(next_line.text) or self.CODE_RE.search(next_line.text):
                            break
                        if next_line.center_y - line.center_y < line.height * 4:
                            issuer_parts.append(next_line.text)
                break

        if issuer_parts:
            issuer_text = " ".join(issuer_parts)
            # Clean up
            issuer_text = re.sub(r"(?i)паспорт\s+выдан\s*", "", issuer_text)
            issuer_text = self.clean_text(issuer_text)
            fields["issuer"] = FieldResult(
                value=issuer_text,
                confidence=0.85 if found else 0.5,
                source="ocr"
            )

    def _extract_issue_date(self, lines: list[OCRLine], fields: dict):
        """Extract passport issue date."""
        result = self.extract_date(lines, context_pattern=r"(?i)дат[аы]\s+выдач")
        if result:
            fields["issue_date"] = FieldResult(
                value=result[0], confidence=result[1], source="ocr"
            )

    def _extract_department_code(self, lines: list[OCRLine], fields: dict):
        """Extract department code (XXX-XXX)."""
        result = self.find_pattern(lines, r"(\d{3})\s*[-–—.]\s*(\d{3})")
        if result:
            text = result[0]
            # Normalize to XXX-XXX
            digits = re.sub(r"\D", "", text)
            if len(digits) == 6:
                code = f"{digits[:3]}-{digits[3:]}"
                fields["department_code"] = FieldResult(
                    value=code, confidence=result[1], source="ocr"
                )

    def _extract_fio(self, lines: list[OCRLine], fields: dict):
        """Extract last name, first name, patronymic."""
        # Strategy 1: Look for labels
        for label, field_name in [
            (r"(?i)фамили[яю]", "last_name"),
            (r"(?i)им[яю]", "first_name"),
            (r"(?i)отчеств[оа]", "patronymic"),
        ]:
            found = self.find_by_label(lines, label)
            if found:
                value = self.clean_text(found.text)
                # Remove the label itself if present
                value = re.sub(label, "", value, flags=re.IGNORECASE).strip()
                value = re.sub(r"^[:\s]+", "", value)
                if value and len(value) > 1:
                    fields[field_name] = FieldResult(
                        value=value.upper(),
                        confidence=found.confidence,
                        source="ocr"
                    )

        # Strategy 2: If no labels found, look for all-caps Cyrillic lines
        if "last_name" not in fields:
            cyrillic_names = []
            for line in lines:
                text = line.text.strip()
                # All uppercase Cyrillic, 2-30 chars, no digits
                if re.match(r"^[А-ЯЁ\s\-]{2,30}$", text) and not re.search(r"\d", text):
                    cyrillic_names.append(line)

            # Typically: last_name, first_name, patronymic in sequence
            name_fields = ["last_name", "first_name", "patronymic"]
            for i, line in enumerate(cyrillic_names[:3]):
                if name_fields[i] not in fields:
                    fields[name_fields[i]] = FieldResult(
                        value=line.text.strip(),
                        confidence=line.confidence * 0.8,  # Lower confidence for heuristic
                        source="ocr"
                    )

    def _extract_birth_date(self, lines: list[OCRLine], fields: dict):
        """Extract birth date."""
        result = self.extract_date(lines, context_pattern=r"(?i)дат[аы]\s+рожден")
        if result:
            fields["birth_date"] = FieldResult(
                value=result[0], confidence=result[1], source="ocr"
            )
        else:
            # Fallback: find all dates, pick the one that looks like a birth date (year 1940-2010)
            dates = []
            for line in lines:
                for m in self.DATE_RE.finditer(line.text):
                    year = int(m.group(3))
                    if 1940 <= year <= 2015:
                        dates.append((f"{m.group(1)}.{m.group(2)}.{m.group(3)}", line.confidence))
            if dates:
                fields["birth_date"] = FieldResult(
                    value=dates[0][0], confidence=dates[0][1] * 0.7, source="ocr"
                )

    def _extract_birth_place(self, lines: list[OCRLine], fields: dict):
        """Extract birth place."""
        found = self.find_by_label(lines, r"(?i)место\s+рожден")
        if found:
            value = self.clean_text(found.text)
            value = re.sub(r"(?i)место\s+рождения\s*:?\s*", "", value)
            if value:
                fields["birth_place"] = FieldResult(
                    value=value, confidence=found.confidence, source="ocr"
                )

    def _extract_sex(self, lines: list[OCRLine], fields: dict):
        """Extract sex (М/Ж)."""
        for line in lines:
            if re.search(r"(?i)\bмуж\.?\b|\bмужской\b", line.text):
                fields["sex"] = FieldResult(value="М", confidence=line.confidence, source="ocr")
                return
            if re.search(r"(?i)\bжен\.?\b|\bженский\b", line.text):
                fields["sex"] = FieldResult(value="Ж", confidence=line.confidence, source="ocr")
                return

        # Fallback: look for standalone М or Ж
        for line in lines:
            text = line.text.strip()
            if text in ("М", "Ж", "м", "ж", "М.", "Ж."):
                fields["sex"] = FieldResult(
                    value=text[0].upper(), confidence=line.confidence * 0.7, source="ocr"
                )
                return
