"""
DocLens Parser: CIS Country Passports
Supports: Uzbekistan, Kyrgyzstan, Tajikistan, Kazakhstan
Relies heavily on MRZ parsing for reliable extraction.
"""
import re
import numpy as np
from app.parsers.base import BaseDocumentParser
from app.core.ocr_engine import OCRResult, OCRLine
from app.core.orchestrator import FieldResult


class PassportCISParser(BaseDocumentParser):
    """Parser for CIS country passports (international format with MRZ)."""

    # MRZ line patterns
    MRZ_LINE1_RE = re.compile(r"P[A-Z<][A-Z<]{3}[A-Z<]+")
    MRZ_LINE2_RE = re.compile(r"[A-Z0-9<]{9}\d[A-Z<]{3}\d{6}\d[MFX<]\d{6}\d[A-Z0-9<]+\d")

    # Country codes
    COUNTRY_NAMES = {
        "UZB": "Узбекистан",
        "KGZ": "Кыргызстан",
        "TJK": "Таджикистан",
        "KAZ": "Казахстан",
        "TKM": "Туркменистан",
        "AZE": "Азербайджан",
        "ARM": "Армения",
        "GEO": "Грузия",
        "MDA": "Молдова",
        "UKR": "Украина",
        "BLR": "Беларусь",
        "RUS": "Россия",
    }

    def parse(self, ocr_result: OCRResult, image: np.ndarray) -> dict[str, FieldResult]:
        fields: dict[str, FieldResult] = {}

        # Try MRZ first (most reliable for international passports)
        mrz_fields = self._parse_mrz(ocr_result)
        if mrz_fields:
            fields.update(mrz_fields)

        # Supplement with Visual Zone OCR
        vz_fields = self._parse_visual_zone(ocr_result)
        for key, value in vz_fields.items():
            if key not in fields:
                fields[key] = value
            elif isinstance(value, FieldResult) and isinstance(fields[key], FieldResult):
                # If MRZ has lower confidence, prefer Visual Zone
                if value.confidence > fields[key].confidence:
                    fields[key] = value

        return fields

    def _parse_mrz(self, ocr_result: OCRResult) -> dict[str, FieldResult]:
        """Parse Machine Readable Zone."""
        fields = {}

        # Find MRZ lines (bottom of document, all-caps with <)
        mrz_lines = []
        for line in ocr_result.lines:
            text = line.text.replace(" ", "")
            if len(text) >= 30 and re.match(r"^[A-Z0-9<]+$", text):
                mrz_lines.append(text)

        if len(mrz_lines) < 2:
            return fields

        # Take last 2 MRZ lines
        line1 = mrz_lines[-2]
        line2 = mrz_lines[-1]

        # Pad to 44 chars
        line1 = line1.ljust(44, "<")
        line2 = line2.ljust(44, "<")

        try:
            # Line 1: P<CCCNAME<<GIVEN<NAMES<<<...
            doc_type = line1[0]  # P = passport
            country = line1[2:5].replace("<", "")

            name_part = line1[5:]
            parts = name_part.split("<<", 1)
            last_name = parts[0].replace("<", " ").strip()
            first_names = parts[1].replace("<", " ").strip() if len(parts) > 1 else ""

            # Line 2: NUMBER___CCCDOB_SEXP_...
            passport_number = line2[0:9].replace("<", "")
            check_number = line2[9]
            nationality = line2[10:13].replace("<", "")
            birth_date_raw = line2[13:19]  # YYMMDD
            check_birth = line2[19]
            sex = line2[20]
            expiry_raw = line2[21:27]  # YYMMDD
            check_expiry = line2[27]

            # Convert dates
            birth_date = self._mrz_date_to_human(birth_date_raw)
            expiry_date = self._mrz_date_to_human(expiry_raw, future=True)

            # Populate fields
            if last_name:
                fields["last_name"] = FieldResult(
                    value=last_name, confidence=0.95, source="mrz"
                )
            if first_names:
                name_parts = first_names.split()
                if name_parts:
                    fields["first_name"] = FieldResult(
                        value=name_parts[0], confidence=0.95, source="mrz"
                    )
                if len(name_parts) > 1:
                    fields["patronymic"] = FieldResult(
                        value=" ".join(name_parts[1:]), confidence=0.90, source="mrz"
                    )

            if passport_number:
                fields["number"] = FieldResult(
                    value=passport_number, confidence=0.95, source="mrz"
                )

            if country:
                country_name = self.COUNTRY_NAMES.get(country, country)
                fields["country"] = FieldResult(
                    value=country_name, confidence=0.98, source="mrz"
                )
                fields["country_code"] = FieldResult(
                    value=country, confidence=0.98, source="mrz"
                )

            if birth_date:
                fields["birth_date"] = FieldResult(
                    value=birth_date, confidence=0.95, source="mrz"
                )

            if sex in ("M", "F"):
                fields["sex"] = FieldResult(
                    value="М" if sex == "M" else "Ж",
                    confidence=0.98, source="mrz"
                )

            if expiry_date:
                fields["expiry_date"] = FieldResult(
                    value=expiry_date, confidence=0.95, source="mrz"
                )

            if nationality:
                fields["nationality_code"] = FieldResult(
                    value=nationality, confidence=0.95, source="mrz"
                )

        except (IndexError, ValueError) as e:
            pass  # MRZ parsing failed, will fall back to Visual Zone

        return fields

    def _parse_visual_zone(self, ocr_result: OCRResult) -> dict[str, FieldResult]:
        """Parse visual zone (printed text on document)."""
        fields = {}
        lines = ocr_result.lines

        # Look for labeled fields
        label_map = {
            r"(?i)(surname|фамили[яю])": "last_name",
            r"(?i)(given\s*name|имен[аяи]|им[яю])": "first_name",
            r"(?i)(patronymic|отчеств)": "patronymic",
            r"(?i)(date\s*of\s*birth|дата\s*рожд)": "birth_date",
            r"(?i)(place\s*of\s*birth|место\s*рожд)": "birth_place",
            r"(?i)(date\s*of\s*issue|дата\s*выдач)": "issue_date",
            r"(?i)(date\s*of\s*expiry|срок\s*действ|годен)": "expiry_date",
            r"(?i)(passport\s*no|номер\s*паспорт)": "number",
        }

        for pattern, field_name in label_map.items():
            found = self.find_by_label(lines, pattern)
            if found:
                value = self.clean_text(found.text)
                value = re.sub(pattern, "", value, flags=re.IGNORECASE).strip()
                value = re.sub(r"^[:\s/]+", "", value)
                if value and len(value) > 1:
                    fields[field_name] = FieldResult(
                        value=value, confidence=found.confidence * 0.85, source="ocr"
                    )

        return fields

    def _mrz_date_to_human(self, yymmdd: str, future: bool = False) -> str | None:
        """Convert MRZ date (YYMMDD) to DD.MM.YYYY."""
        if len(yymmdd) != 6 or not yymmdd.isdigit():
            return None
        yy = int(yymmdd[:2])
        mm = yymmdd[2:4]
        dd = yymmdd[4:6]

        # Determine century
        if future:
            year = 2000 + yy  # Expiry dates are usually in the future
        else:
            year = 1900 + yy if yy > 30 else 2000 + yy

        return f"{dd}.{mm}.{year}"
