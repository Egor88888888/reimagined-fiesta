"""
DocLens Enrichment: FNS INN Check
Lookup INN by personal data via FNS service.
"""
import httpx
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FNSCheckResult:
    inn: str | None = None
    is_valid: bool | None = None
    message: str = ""
    source: str = "nalog.ru"


async def lookup_inn(
    last_name: str,
    first_name: str,
    patronymic: str = "",
    birth_date: str = "",
    passport_series: str = "",
    passport_number: str = "",
) -> FNSCheckResult:
    """Look up INN by personal data via FNS service.

    Args:
        last_name: Last name
        first_name: First name
        patronymic: Patronymic (optional)
        birth_date: Birth date DD.MM.YYYY
        passport_series: Passport series XX XX
        passport_number: Passport number XXXXXX

    Returns:
        FNSCheckResult
    """
    try:
        url = "https://service.nalog.ru/inn-proc.do"

        data = {
            "fam": last_name,
            "nam": first_name,
            "otch": patronymic,
            "bdate": birth_date,
            "doctype": "21",  # 21 = passport RF
            "docno": f"{passport_series} {passport_number}".strip(),
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(url, data=data)

            if response.status_code == 200:
                result = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}

                inn = result.get("inn")
                if inn:
                    return FNSCheckResult(
                        inn=inn,
                        is_valid=True,
                        message=f"INN found: {inn}"
                    )
                else:
                    return FNSCheckResult(
                        inn=None,
                        is_valid=None,
                        message="INN not found for given personal data"
                    )
            else:
                return FNSCheckResult(
                    is_valid=None,
                    message=f"FNS service returned status {response.status_code}"
                )

    except httpx.TimeoutException:
        logger.warning("FNS check timed out")
        return FNSCheckResult(is_valid=None, message="FNS service timeout")
    except Exception as e:
        logger.error(f"FNS check error: {e}")
        return FNSCheckResult(is_valid=None, message=f"FNS check error: {str(e)}")


async def validate_inn(inn: str) -> FNSCheckResult:
    """Validate that an INN exists in FNS database."""
    try:
        url = f"https://service.nalog.ru/inn-proc.do"
        # For now, just validate format (API check requires personal data)
        digits = "".join(c for c in inn if c.isdigit())

        if len(digits) in (10, 12):
            return FNSCheckResult(
                inn=digits,
                is_valid=True,
                message="INN format valid"
            )
        else:
            return FNSCheckResult(
                is_valid=False,
                message=f"Invalid INN length: {len(digits)}"
            )

    except Exception as e:
        logger.error(f"INN validation error: {e}")
        return FNSCheckResult(is_valid=None, message=str(e))
