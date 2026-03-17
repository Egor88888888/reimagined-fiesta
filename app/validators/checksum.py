"""
DocLens Validators
Checksum validation for Russian documents.
"""
import re
from app.models.database import DocumentType


def validate_document_fields(doc_type: DocumentType, fields: dict) -> dict:
    """Validate extracted fields based on document type.

    Returns:
        Dict of field_name -> {"valid": bool, "message": str}
    """
    validators = {
        DocumentType.PASSPORT_RF: _validate_passport_rf,
        DocumentType.SNILS: _validate_snils,
        DocumentType.INN: _validate_inn,
        DocumentType.DRIVER_LICENSE: _validate_driver_license,
        DocumentType.PASSPORT_CIS: _validate_passport_cis,
    }

    validator = validators.get(doc_type)
    if validator:
        return validator(fields)
    return {}


# ============================================================
# PASSPORT RF
# ============================================================

def _validate_passport_rf(fields: dict) -> dict:
    results = {}

    # Series format: XX XX (2 digits, space, 2 digits)
    series = _get_value(fields, "series")
    if series:
        clean = re.sub(r"\s", "", series)
        if re.match(r"^\d{4}$", clean):
            results["series"] = {"valid": True, "message": "Format OK"}
        else:
            results["series"] = {"valid": False, "message": "Series must be 4 digits (XX XX)"}

    # Number format: 6 digits
    number = _get_value(fields, "number")
    if number:
        clean = re.sub(r"\s", "", number)
        if re.match(r"^\d{6}$", clean):
            results["number"] = {"valid": True, "message": "Format OK"}
        else:
            results["number"] = {"valid": False, "message": "Number must be 6 digits"}

    # Department code: XXX-XXX
    code = _get_value(fields, "department_code")
    if code:
        clean = re.sub(r"[\s\-]", "", code)
        if re.match(r"^\d{6}$", clean):
            results["department_code"] = {"valid": True, "message": "Format OK"}
        else:
            results["department_code"] = {"valid": False, "message": "Code must be XXX-XXX"}

    # Dates
    for field_name in ["birth_date", "issue_date"]:
        date = _get_value(fields, field_name)
        if date:
            results[field_name] = _validate_date(date)

    return results


# ============================================================
# SNILS
# ============================================================

def _validate_snils(fields: dict) -> dict:
    results = {}

    snils = _get_value(fields, "snils_number")
    if snils:
        digits = re.sub(r"\D", "", snils)
        if len(digits) != 11:
            results["snils_number"] = {
                "valid": False,
                "message": f"SNILS must be 11 digits, got {len(digits)}"
            }
        else:
            # Checksum validation
            number = digits[:9]
            control = int(digits[9:11])

            # Numbers <= 001-001-998 have checksums
            num_val = int(number)
            if num_val <= 1001998:
                results["snils_number"] = {"valid": True, "message": "Below checksum range, format OK"}
            else:
                total = sum(int(d) * (9 - i) for i, d in enumerate(number))

                if total < 100:
                    expected = total
                elif total in (100, 101):
                    expected = 0
                else:
                    remainder = total % 101
                    expected = 0 if remainder in (100, 101) else remainder

                if control == expected:
                    results["snils_number"] = {"valid": True, "message": "Checksum valid"}
                else:
                    results["snils_number"] = {
                        "valid": False,
                        "message": f"Checksum invalid: expected {expected:02d}, got {control:02d}"
                    }

    return results


# ============================================================
# INN
# ============================================================

def _validate_inn(fields: dict) -> dict:
    results = {}

    inn = _get_value(fields, "inn_number")
    if inn:
        digits = re.sub(r"\D", "", inn)

        if len(digits) == 12:
            # Individual INN (12 digits)
            d = [int(c) for c in digits]
            w11 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
            w12 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]

            check_11 = sum(w * d[i] for i, w in enumerate(w11)) % 11 % 10
            check_12 = sum(w * d[i] for i, w in enumerate(w12)) % 11 % 10

            if d[10] == check_11 and d[11] == check_12:
                results["inn_number"] = {"valid": True, "message": "Checksum valid (individual)"}
            else:
                results["inn_number"] = {
                    "valid": False,
                    "message": "INN checksum invalid"
                }

        elif len(digits) == 10:
            # Legal entity INN (10 digits)
            d = [int(c) for c in digits]
            w10 = [2, 4, 10, 3, 5, 9, 4, 6, 8]
            check_10 = sum(w * d[i] for i, w in enumerate(w10)) % 11 % 10

            if d[9] == check_10:
                results["inn_number"] = {"valid": True, "message": "Checksum valid (legal entity)"}
            else:
                results["inn_number"] = {
                    "valid": False,
                    "message": "INN checksum invalid"
                }
        else:
            results["inn_number"] = {
                "valid": False,
                "message": f"INN must be 10 or 12 digits, got {len(digits)}"
            }

    return results


# ============================================================
# DRIVER LICENSE
# ============================================================

def _validate_driver_license(fields: dict) -> dict:
    results = {}

    number = _get_value(fields, "number")
    if number:
        digits = re.sub(r"\D", "", number)
        if len(digits) == 10:
            results["number"] = {"valid": True, "message": "Format OK (10 digits)"}
        else:
            results["number"] = {"valid": False, "message": f"Expected 10 digits, got {len(digits)}"}

    categories = _get_value(fields, "categories")
    if categories:
        valid_cats = {"A", "A1", "B", "B1", "BE", "C", "C1", "CE", "C1E", "D", "D1", "DE", "D1E", "M", "Tm", "Tb"}
        found = set(re.findall(r"[ABCDEM][1]?[E]?", categories))
        if found:
            results["categories"] = {"valid": True, "message": f"Categories: {', '.join(found)}"}

    for field_name in ["birth_date", "issue_date", "expiry_date"]:
        date = _get_value(fields, field_name)
        if date:
            results[field_name] = _validate_date(date)

    return results


# ============================================================
# PASSPORT CIS
# ============================================================

def _validate_passport_cis(fields: dict) -> dict:
    results = {}

    for field_name in ["birth_date", "issue_date", "expiry_date"]:
        date = _get_value(fields, field_name)
        if date:
            results[field_name] = _validate_date(date)

    return results


# ============================================================
# Helpers
# ============================================================

def _get_value(fields: dict, key: str) -> str | None:
    """Get field value from nested dict."""
    field = fields.get(key)
    if isinstance(field, dict):
        return field.get("value")
    return None


def _validate_date(date_str: str) -> dict:
    """Validate date format DD.MM.YYYY."""
    m = re.match(r"^(\d{2})\.(\d{2})\.(\d{4})$", date_str)
    if not m:
        return {"valid": False, "message": f"Invalid date format: {date_str}"}

    day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= month <= 12):
        return {"valid": False, "message": f"Invalid month: {month}"}
    if not (1 <= day <= 31):
        return {"valid": False, "message": f"Invalid day: {day}"}
    if not (1900 <= year <= 2100):
        return {"valid": False, "message": f"Invalid year: {year}"}

    return {"valid": True, "message": "Date format OK"}
